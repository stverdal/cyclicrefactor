"""Dependency Analyzer Agent - Builds dependency graph from source directory.

This agent scans a project directory and builds a complete dependency graph
by analyzing import statements in TypeScript/JavaScript files.

This is an optional pre-pipeline step that enables automatic cycle detection
without requiring the user to provide cycle information upfront.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from .agent_base import Agent, AgentResult
from models.schemas import DependencyGraph, AnalysisResult
from utils.ts_dependency_analyzer import (
    analyze_project, 
    AnalyzerConfig, 
    get_project_info,
    check_madge_available,
    check_node_available,
)
from utils.logging import get_logger

logger = get_logger("dependency_analyzer")


class DependencyAnalyzerAgent(Agent):
    """Analyzes a project directory to build a dependency graph.
    
    This agent supports TypeScript and JavaScript projects, using:
    - madge (npm package) for accurate dependency extraction
    - Fallback regex-based parsing when madge is unavailable
    
    Usage:
        agent = DependencyAnalyzerAgent()
        result = agent.run({
            "project_dir": "/path/to/project",
            "extensions": ["ts", "tsx"],
            "exclude_patterns": ["node_modules", "dist"],
        })
        
        if result.success:
            graph = result.data["graph"]
            print(f"Found {len(graph.nodes)} files")
    """
    
    name = "dependency_analyzer"
    version = "1.0"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dependency analyzer agent.
        
        Args:
            config: Optional configuration dict with keys:
                - prefer_madge: Whether to prefer madge over regex (default True)
                - default_extensions: Default file extensions to analyze
                - default_excludes: Default patterns to exclude
        """
        self.config = config or {}
        self.prefer_madge = self.config.get("prefer_madge", True)
        self.default_extensions = self.config.get(
            "default_extensions", 
            ["ts", "tsx", "js", "jsx"]
        )
        self.default_excludes = self.config.get(
            "default_excludes",
            ["node_modules", "dist", "build", ".git", "__tests__", "*.test.*", "*.spec.*"]
        )
    
    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """Analyze a project directory for dependencies.
        
        Args:
            input_data: Dict containing:
                - project_dir (required): Path to the project root
                - extensions (optional): List of file extensions to include
                - exclude_patterns (optional): List of patterns to exclude
                - include_external (optional): Include node_modules deps (default False)
                - tsconfig_path (optional): Path to tsconfig.json
                
        Returns:
            AgentResult with:
                - success: True if analysis completed
                - data: Contains 'graph' (DependencyGraph) and 'project_info'
                - error: Error message if failed
        """
        start_time = time.time()
        
        # Validate input
        project_dir = input_data.get("project_dir")
        if not project_dir:
            return AgentResult(
                status="error",
                output={"error": "project_dir is required"},
            )
        
        project_path = Path(project_dir).resolve()
        if not project_path.exists():
            return AgentResult(
                status="error",
                output={"error": f"Project directory does not exist: {project_dir}"},
            )
        
        if not project_path.is_dir():
            return AgentResult(
                status="error",
                output={"error": f"Path is not a directory: {project_dir}"},
            )
        
        # Build analyzer config
        config = AnalyzerConfig(
            extensions=input_data.get("extensions", self.default_extensions),
            exclude_patterns=input_data.get("exclude_patterns", self.default_excludes),
            include_external=input_data.get("include_external", False),
            tsconfig_path=input_data.get("tsconfig_path"),
            webpack_config=input_data.get("webpack_config"),
        )
        
        logger.info(f"Analyzing project: {project_path}")
        logger.info(f"Extensions: {config.extensions}")
        logger.info(f"Excludes: {config.exclude_patterns}")
        logger.debug(f"Include external: {config.include_external}")
        logger.debug(f"TSConfig path: {config.tsconfig_path or 'auto-detect'}")
        logger.debug(f"Prefer madge: {self.prefer_madge}")
        
        # Check tool availability
        tools_info = {
            "node_available": check_node_available(),
            "madge_available": check_madge_available(),
        }
        
        logger.info(f"Tool availability: Node.js={tools_info['node_available']}, madge={tools_info['madge_available']}")
        
        if not tools_info["node_available"]:
            logger.warning("Node.js not found - using regex-based analysis (less accurate)")
        elif not tools_info["madge_available"]:
            logger.warning("madge not found - using regex-based analysis. Install with: npm install -g madge")
        
        try:
            # Get project metadata
            project_info = get_project_info(str(project_path))
            
            # Run analysis
            graph = analyze_project(
                str(project_path),
                config=config,
                prefer_madge=self.prefer_madge,
            )
            
            elapsed = time.time() - start_time
            
            logger.info(f"Analysis complete in {elapsed:.2f}s")
            logger.info(f"Found {len(graph.nodes)} files, {len(graph.edges)} dependencies")
            
            return AgentResult(
                status="success",
                output={
                    "graph": graph,
                    "project_info": project_info,
                    "tools": tools_info,
                    "analysis_time_seconds": elapsed,
                },
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__}: {e}")
            logger.debug(f"Exception details", exc_info=True)
            return AgentResult(
                status="error",
                output={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "tools": tools_info,
                    "analysis_time_seconds": time.time() - start_time,
                },
            )
    
    def check_prerequisites(self) -> Dict[str, Any]:
        """Check if required tools are available.
        
        Returns:
            Dict with tool availability info and recommendations
        """
        node_available = check_node_available()
        madge_available = check_madge_available()
        
        result = {
            "ready": True,
            "node_available": node_available,
            "madge_available": madge_available,
            "recommendations": [],
        }
        
        if not node_available:
            result["recommendations"].append(
                "Install Node.js for better dependency analysis: https://nodejs.org"
            )
        elif not madge_available:
            result["recommendations"].append(
                "Install madge for accurate TypeScript analysis: npm install -g madge"
            )
        
        # Regex fallback is always available
        result["fallback_available"] = True
        
        return result
