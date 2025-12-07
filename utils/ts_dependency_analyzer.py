"""TypeScript/JavaScript dependency analyzer using madge.

This module provides dependency graph extraction for TypeScript and JavaScript
projects using the 'madge' npm package as the primary backend, with fallback
to a simple regex-based parser.

Requirements:
    - Node.js and npm installed
    - madge: Install globally with `npm install -g madge`
    - For TypeScript: typescript package in project or globally

Usage:
    from utils.ts_dependency_analyzer import analyze_project
    
    graph = analyze_project("/path/to/project")
    print(f"Found {len(graph.nodes)} files with {len(graph.edges)} dependencies")
"""

import json
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("ts_dependency_analyzer")


@dataclass
class AnalyzerConfig:
    """Configuration for dependency analysis."""
    extensions: List[str] = field(default_factory=lambda: ["ts", "tsx", "js", "jsx"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "node_modules", 
        "dist", 
        "build", 
        ".git",
        "__tests__",
        "*.test.*",
        "*.spec.*",
    ])
    include_external: bool = False  # Include node_modules dependencies
    tsconfig_path: Optional[str] = None  # Path to tsconfig.json
    webpack_config: Optional[str] = None  # Path to webpack.config.js
    follow_type_imports: bool = True  # Include "import type" statements
    max_depth: Optional[int] = None  # Max directory depth to scan


def check_madge_available() -> bool:
    """Check if madge is installed and available."""
    try:
        logger.debug("Checking madge availability via npx...")
        result = subprocess.run(
            ["npx", "madge", "--version"],
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout for first-time npx downloads
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.debug(f"madge available, version: {version}")
            return True
        else:
            logger.debug(f"madge check failed: exit code {result.returncode}, stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("madge availability check timed out (>60s) - npx may be downloading")
        return False
    except FileNotFoundError:
        logger.debug("npx not found - Node.js may not be installed")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error checking madge: {type(e).__name__}: {e}")
        return False


def check_node_available() -> bool:
    """Check if Node.js is installed."""
    node_path = shutil.which("node")
    if node_path:
        logger.debug(f"Node.js found at: {node_path}")
        return True
    else:
        logger.debug("Node.js not found in PATH")
        return False


def analyze_with_madge(
    project_dir: str,
    config: Optional[AnalyzerConfig] = None
) -> Dict[str, Any]:
    """Analyze project dependencies using madge.
    
    Args:
        project_dir: Path to the project root
        config: Optional analyzer configuration
        
    Returns:
        Dict with 'graph' (adjacency list) and 'circular' (list of cycles)
        
    Raises:
        RuntimeError: If madge is not available or fails
    """
    if config is None:
        config = AnalyzerConfig()
    
    project_path = Path(project_dir).resolve()
    
    if not project_path.exists():
        raise ValueError(f"Project directory does not exist: {project_dir}")
    
    # Build madge command
    cmd = [
        "npx", "madge",
        "--json",  # Output as JSON
        "--extensions", ",".join(config.extensions),
    ]
    
    # Add tsconfig if available
    tsconfig = config.tsconfig_path
    if not tsconfig:
        # Try to find tsconfig.json
        default_tsconfig = project_path / "tsconfig.json"
        if default_tsconfig.exists():
            tsconfig = str(default_tsconfig)
    
    if tsconfig:
        cmd.extend(["--ts-config", tsconfig])
    
    # Add webpack config if available
    if config.webpack_config:
        cmd.extend(["--webpack-config", config.webpack_config])
    
    # Add the project directory
    cmd.append(str(project_path))
    
    logger.info(f"Running madge: {' '.join(cmd)}")
    logger.debug(f"Working directory: {project_path}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for large projects
            cwd=str(project_path),
        )
        elapsed = time.time() - start_time
        logger.debug(f"Madge completed in {elapsed:.1f}s, exit code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"Madge failed (exit code {result.returncode})")
            logger.error(f"Madge stderr: {result.stderr[:500]}")
            if result.stdout:
                logger.debug(f"Madge stdout: {result.stdout[:500]}")
            raise RuntimeError(f"Madge analysis failed: {result.stderr}")
        
        # Parse the JSON output (adjacency list format)
        graph_data = json.loads(result.stdout)
        
        # Also get circular dependencies
        circular_cmd = cmd.copy()
        circular_cmd.insert(2, "--circular")
        
        logger.debug("Running madge --circular to detect cycles...")
        circular_result = subprocess.run(
            circular_cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_path),
        )
        
        circular_data = []
        logger.debug(f"Circular detection exit code: {circular_result.returncode}")
        if circular_result.returncode == 0 and circular_result.stdout.strip():
            try:
                circular_data = json.loads(circular_result.stdout)
                logger.info(f"Madge detected {len(circular_data)} circular dependency chains")
            except json.JSONDecodeError:
                # Madge might output text format for circular deps
                logger.warning(f"Circular output not JSON (may be text format): {circular_result.stdout[:200]}")
        elif circular_result.returncode != 0:
            logger.warning(f"Circular detection returned non-zero: {circular_result.stderr[:200]}")
        
        return {
            "graph": graph_data,
            "circular": circular_data,
            "tool": "madge",
        }
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Madge analysis timed out (>120s)")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse madge output: {e}")


def analyze_with_regex(
    project_dir: str,
    config: Optional[AnalyzerConfig] = None
) -> Dict[str, Any]:
    """Fallback analyzer using regex-based import extraction.
    
    This is less accurate than madge but doesn't require Node.js.
    
    Args:
        project_dir: Path to the project root
        config: Optional analyzer configuration
        
    Returns:
        Dict with 'graph' (adjacency list) and 'circular' (empty list)
    """
    if config is None:
        config = AnalyzerConfig()
    
    project_path = Path(project_dir).resolve()
    
    # Collect all source files
    source_files = []
    for ext in config.extensions:
        source_files.extend(project_path.rglob(f"*.{ext}"))
    
    # Filter out excluded patterns
    def is_excluded(path: Path) -> bool:
        path_str = str(path)
        for pattern in config.exclude_patterns:
            if pattern in path_str:
                return True
        return False
    
    excluded_count = len(source_files)
    source_files = [f for f in source_files if not is_excluded(f)]
    excluded_count = excluded_count - len(source_files)
    
    logger.info(f"Regex analyzer: Found {len(source_files)} source files ({excluded_count} excluded)")
    if len(source_files) == 0:
        logger.warning("No source files found - check extensions and exclude patterns")
        logger.debug(f"Extensions searched: {config.extensions}")
        logger.debug(f"Exclude patterns: {config.exclude_patterns}")
    
    # Build adjacency list
    graph: Dict[str, List[str]] = {}
    files_with_imports = 0
    total_imports_found = 0
    
    # Import patterns
    patterns = [
        # ES6: import X from 'Y' or import { X } from 'Y'
        re.compile(r"import\s+(?:type\s+)?(?:[^'\"]+\s+from\s+)?['\"]([^'\"]+)['\"]"),
        # CommonJS: require('X')
        re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        # Dynamic: import('X')
        re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        # Re-export: export * from 'X' or export { X } from 'Y'
        re.compile(r"export\s+(?:\*|{[^}]+})\s+from\s+['\"]([^'\"]+)['\"]"),
    ]
    
    for source_file in source_files:
        rel_path = str(source_file.relative_to(project_path))
        # Normalize path separators
        rel_path = rel_path.replace("\\", "/")
        
        try:
            content = source_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {source_file}: {e}")
            continue
        
        imports = set()
        unresolved_imports = []
        for pattern in patterns:
            for match in pattern.finditer(content):
                import_path = match.group(1)
                # Only include relative imports (internal dependencies)
                if import_path.startswith("."):
                    # Resolve the relative import
                    resolved = resolve_relative_import(
                        import_path, 
                        source_file, 
                        project_path,
                        config.extensions
                    )
                    if resolved:
                        imports.add(resolved)
                    else:
                        unresolved_imports.append(import_path)
                elif not config.include_external:
                    # Skip external imports
                    continue
                else:
                    imports.add(import_path)
        
        if imports:
            files_with_imports += 1
            total_imports_found += len(imports)
        if unresolved_imports:
            logger.debug(f"{rel_path}: {len(unresolved_imports)} unresolved imports: {unresolved_imports[:3]}")
        
        graph[rel_path] = sorted(list(imports))
    
    logger.info(f"Regex analysis complete: {files_with_imports}/{len(source_files)} files have imports, {total_imports_found} total imports")
    
    return {
        "graph": graph,
        "circular": [],  # Regex analyzer doesn't detect cycles
        "tool": "regex",
    }


def resolve_relative_import(
    import_path: str,
    from_file: Path,
    project_root: Path,
    extensions: List[str]
) -> Optional[str]:
    """Resolve a relative import path to an actual file.
    
    Handles:
    - Direct file imports: './foo' -> './foo.ts'
    - Index imports: './foo' -> './foo/index.ts'
    - Extension-less imports
    """
    from_dir = from_file.parent
    
    # Resolve the relative path
    target = (from_dir / import_path).resolve()
    
    # Try with each extension
    for ext in extensions:
        # Try direct file
        candidate = target.with_suffix(f".{ext}")
        if candidate.exists():
            try:
                return str(candidate.relative_to(project_root)).replace("\\", "/")
            except ValueError:
                continue
        
        # Try as directory with index file
        index_candidate = target / f"index.{ext}"
        if index_candidate.exists():
            try:
                return str(index_candidate.relative_to(project_root)).replace("\\", "/")
            except ValueError:
                continue
    
    # If the path already has an extension, try it directly
    if target.exists():
        try:
            return str(target.relative_to(project_root)).replace("\\", "/")
        except ValueError:
            pass
    
    return None


def convert_to_dependency_graph(
    analysis_result: Dict[str, Any],
    project_dir: str,
    config: Optional[AnalyzerConfig] = None
) -> "DependencyGraph":
    """Convert madge/regex output to DependencyGraph model.
    
    Args:
        analysis_result: Output from analyze_with_madge or analyze_with_regex
        project_dir: Project root directory
        config: Analyzer configuration
        
    Returns:
        DependencyGraph model instance
    """
    from models.schemas import DependencyGraph, ImportInfo
    
    graph_data = analysis_result.get("graph", {})
    tool_used = analysis_result.get("tool", "unknown")
    
    logger.debug(f"Converting {tool_used} output to DependencyGraph ({len(graph_data)} source files)")
    
    nodes = set()
    edges = []
    imports = []
    external_count = 0
    
    for source_file, dependencies in graph_data.items():
        nodes.add(source_file)
        
        for dep in dependencies:
            nodes.add(dep)
            edges.append([source_file, dep])
            
            # Determine if external
            is_external = not dep.startswith(".") and "/" not in dep.split("/")[0]
            if is_external:
                external_count += 1
            
            imports.append(ImportInfo(
                source_file=source_file,
                imported_from=dep,
                resolved_path=dep if not is_external else None,
                import_type="es6",  # Default assumption
                is_external=is_external,
            ))
    
    logger.debug(f"Graph conversion complete: {len(nodes)} nodes, {len(edges)} edges, {external_count} external imports")
    
    return DependencyGraph(
        root_dir=project_dir,
        nodes=sorted(list(nodes)),
        edges=edges,
        imports=imports,
        metadata={
            "tool": tool_used,
            "config": config.__dict__ if config else {},
        }
    )


def analyze_project(
    project_dir: str,
    config: Optional[AnalyzerConfig] = None,
    prefer_madge: bool = True
) -> "DependencyGraph":
    """Analyze a TypeScript/JavaScript project for dependencies.
    
    This is the main entry point. It will use madge if available,
    falling back to regex-based analysis.
    
    Args:
        project_dir: Path to the project root
        config: Optional analyzer configuration
        prefer_madge: If True, prefer madge over regex (default True)
        
    Returns:
        DependencyGraph with all dependencies mapped
        
    Example:
        >>> graph = analyze_project("/path/to/my-react-app")
        >>> print(f"Files: {len(graph.nodes)}, Imports: {len(graph.edges)}")
        Files: 150, Imports: 423
    """
    if config is None:
        config = AnalyzerConfig()
    
    logger.info(f"Analyzing project: {project_dir}")
    
    # Check if madge is available
    use_madge = prefer_madge and check_madge_available()
    
    if use_madge:
        logger.info("Using madge for dependency analysis")
        try:
            result = analyze_with_madge(project_dir, config)
        except RuntimeError as e:
            logger.warning(f"Madge failed, falling back to regex: {e}")
            result = analyze_with_regex(project_dir, config)
    else:
        if prefer_madge:
            logger.warning("Madge not available, using regex-based analysis")
            logger.info("Install madge for better results: npm install -g madge")
        result = analyze_with_regex(project_dir, config)
    
    graph = convert_to_dependency_graph(result, project_dir, config)
    
    logger.info(f"Analysis complete: {len(graph.nodes)} files, {len(graph.edges)} dependencies")
    
    return graph


def get_project_info(project_dir: str) -> Dict[str, Any]:
    """Get metadata about a project (package.json, tsconfig, etc.)."""
    project_path = Path(project_dir).resolve()
    info = {
        "has_package_json": False,
        "has_tsconfig": False,
        "name": project_path.name,
        "type": "unknown",
    }
    
    # Check package.json
    package_json = project_path / "package.json"
    if package_json.exists():
        info["has_package_json"] = True
        try:
            with open(package_json) as f:
                pkg = json.load(f)
                info["name"] = pkg.get("name", info["name"])
                info["dependencies"] = list(pkg.get("dependencies", {}).keys())
                info["devDependencies"] = list(pkg.get("devDependencies", {}).keys())
        except Exception:
            pass
    
    # Check tsconfig.json
    tsconfig = project_path / "tsconfig.json"
    if tsconfig.exists():
        info["has_tsconfig"] = True
        info["type"] = "typescript"
        try:
            with open(tsconfig) as f:
                ts = json.load(f)
                info["ts_paths"] = ts.get("compilerOptions", {}).get("paths", {})
        except Exception:
            pass
    else:
        info["type"] = "javascript"
    
    return info
