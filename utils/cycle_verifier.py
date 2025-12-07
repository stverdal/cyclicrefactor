"""Cycle verification utilities.

This module provides functions to verify whether a refactoring proposal
actually breaks the cyclic dependency by re-analyzing the import/dependency
graph after applying the proposed patches.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re

from utils.logging import get_logger

logger = get_logger("cycle_verifier")


@dataclass
class CycleVerificationResult:
    """Result of verifying whether a cycle was broken."""
    is_broken: bool
    remaining_cycles: List[List[str]]  # List of cycles still present
    removed_edges: List[Tuple[str, str]]  # Edges that were removed
    added_edges: List[Tuple[str, str]]  # New edges that were added
    analysis: str  # Human-readable analysis
    confidence: float  # 0.0-1.0 confidence in the analysis


def extract_imports_python(content: str) -> Set[str]:
    """Extract imported module names from Python code."""
    imports = set()
    
    # import X, import X as Y
    for match in re.finditer(r'^\s*import\s+([\w.]+)', content, re.MULTILINE):
        imports.add(match.group(1).split('.')[0])
    
    # from X import Y, from X.Y import Z
    for match in re.finditer(r'^\s*from\s+([\w.]+)\s+import', content, re.MULTILINE):
        imports.add(match.group(1).split('.')[0])
    
    return imports


def extract_imports_csharp(content: str) -> Set[str]:
    """Extract namespace references from C# code."""
    imports = set()
    
    # using Namespace;
    for match in re.finditer(r'^\s*using\s+([\w.]+)\s*;', content, re.MULTILINE):
        # Get the root namespace
        imports.add(match.group(1).split('.')[0])
    
    return imports


def extract_type_references(content: str, known_types: Set[str]) -> Set[str]:
    """Find references to known types in code content."""
    refs = set()
    for type_name in known_types:
        # Look for type name as identifier (not in comments or strings - simplified)
        pattern = rf'\b{re.escape(type_name)}\b'
        if re.search(pattern, content):
            refs.add(type_name)
    return refs


def extract_imports(content: str, file_path: str) -> Set[str]:
    """Extract imports based on file type."""
    ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
    
    if ext == 'py':
        return extract_imports_python(content)
    elif ext == 'cs':
        return extract_imports_csharp(content)
    elif ext in ('ts', 'tsx', 'js', 'jsx'):
        # TypeScript/JavaScript - simplified
        imports = set()
        for match in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", content):
            module = match.group(1)
            # Handle relative imports
            if module.startswith('.'):
                imports.add(module.split('/')[-1].replace('.', ''))
            else:
                imports.add(module.split('/')[0])
        return imports
    else:
        return set()


def _match_file_to_node(file_node: str, cycle_nodes: List[str]) -> Optional[str]:
    """Match a file name to a cycle node with precise matching.
    
    Matching priority:
    1. Exact match (case-insensitive)
    2. Exact match without 'I' prefix (interface files like IFoo -> Foo)
    3. Node is a suffix of file name with word boundary (FooService matches Foo)
    4. File is a suffix of node name with word boundary
    
    Returns:
        Matching cycle node name or None
    """
    file_lower = file_node.lower()
    
    for cn in cycle_nodes:
        cn_lower = cn.lower()
        
        # 1. Exact match
        if file_lower == cn_lower:
            return cn
        
        # 2. Interface pattern: IFoo matches Foo
        if file_lower.startswith('i') and file_lower[1:] == cn_lower:
            return cn
        if cn_lower.startswith('i') and cn_lower[1:] == file_lower:
            return cn
        
        # 3. Node is exact suffix with word boundary (e.g., SensorStore contains Sensor? No - too loose)
        # Be more strict: only match if file_node ends with cn OR cn ends with file_node
        if file_lower.endswith(cn_lower) and (len(file_lower) == len(cn_lower) or 
            not file_lower[-(len(cn_lower)+1)].isalnum()):
            return cn
        if cn_lower.endswith(file_lower) and (len(cn_lower) == len(file_lower) or
            not cn_lower[-(len(file_lower)+1)].isalnum()):
            return cn
    
    return None


def build_dependency_graph(
    files: List[Dict[str, str]],
    cycle_nodes: List[str],
    node_file_map: Optional[Dict[str, str]] = None
) -> Dict[str, Set[str]]:
    """Build a dependency graph from file contents.
    
    Args:
        files: List of {"path": str, "content": str}
        cycle_nodes: List of node names in the cycle
        node_file_map: Optional explicit mapping of node names to file paths
        
    Returns:
        Dict mapping node name to set of nodes it depends on
    """
    graph = {node: set() for node in cycle_nodes}
    
    # Build reverse mapping: file path -> node name
    file_to_node = {}
    if node_file_map:
        # Use explicit mapping (reverse it)
        for node, path in node_file_map.items():
            file_to_node[path] = node
            # Also add with basename for partial matching
            basename = path.split('/')[-1].split('\\')[-1]
            file_to_node[basename] = node
    
    # Map file paths to node names (fallback using basename without extension)
    path_to_node = {}
    for f in files:
        path = f.get("path", "")
        
        # First check explicit mapping
        if path in file_to_node:
            path_to_node[path] = file_to_node[path]
        else:
            # Fallback to basename without extension
            basename = path.split('/')[-1].split('\\')[-1]
            if basename in file_to_node:
                path_to_node[path] = file_to_node[basename]
            else:
                name_without_ext = basename.rsplit('.', 1)[0] if '.' in basename else basename
                path_to_node[path] = name_without_ext
    
    # Also map cycle nodes to themselves
    node_set = set(cycle_nodes)
    
    for f in files:
        path = f.get("path", "")
        content = f.get("content", "")
        
        # Determine which node this file represents
        file_node = path_to_node.get(path)
        if not file_node:
            continue
            
        # If explicit mapping was used, file_node is already the cycle node
        if node_file_map and file_node in cycle_nodes:
            source_node = file_node
        else:
            # Find matching cycle node using improved matching
            source_node = _match_file_to_node(file_node, cycle_nodes)
        
        if not source_node:
            continue
            
        # Extract imports and references
        imports = extract_imports(content, path)
        type_refs = extract_type_references(content, node_set)
        
        # Find dependencies to other cycle nodes
        for target_node in cycle_nodes:
            if target_node == source_node:
                continue
                
            # Check if this file imports/references the target
            target_lower = target_node.lower()
            
            # Check in imports
            for imp in imports:
                if target_lower in imp.lower():
                    graph[source_node].add(target_node)
                    break
            
            # Check in type references
            if target_node in type_refs:
                graph[source_node].add(target_node)
    
    return graph


def find_cycles_in_graph(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find all cycles in a directed graph using Tarjan's algorithm for SCCs.
    
    Args:
        graph: Dict mapping node to set of nodes it points to
        
    Returns:
        List of cycles (each cycle is a list of nodes)
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    def strongconnect(node):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True
        
        for successor in graph.get(node, set()):
            if successor not in index:
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif on_stack.get(successor, False):
                lowlinks[node] = min(lowlinks[node], index[successor])
        
        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            # Only include SCCs with more than one node (actual cycles)
            if len(scc) > 1:
                sccs.append(scc)
    
    for node in graph:
        if node not in index:
            strongconnect(node)
    
    return sccs


def verify_cycle_broken(
    original_files: List[Dict[str, str]],
    patched_files: List[Dict[str, str]],
    cycle_nodes: List[str],
    original_edges: List[List[str]],
    node_file_map: Optional[Dict[str, str]] = None
) -> CycleVerificationResult:
    """Verify whether the proposed patches break the cycle.
    
    Args:
        original_files: Original file contents [{"path": str, "content": str}]
        patched_files: Patched file contents [{"path": str, "content": str}]
        cycle_nodes: List of node names in the cycle
        original_edges: Original edges [[from, to], ...]
        node_file_map: Optional explicit mapping of node names to file paths
        
    Returns:
        CycleVerificationResult with analysis
    """
    logger.info(f"Verifying cycle with {len(cycle_nodes)} nodes")
    if node_file_map:
        logger.info(f"Using explicit node-file mapping with {len(node_file_map)} entries")
    
    # Build graphs for original and patched
    original_graph = build_dependency_graph(original_files, cycle_nodes, node_file_map)
    patched_graph = build_dependency_graph(patched_files, cycle_nodes, node_file_map)
    
    # Find edges that changed
    removed_edges = []
    added_edges = []
    
    for node in cycle_nodes:
        orig_deps = original_graph.get(node, set())
        patch_deps = patched_graph.get(node, set())
        
        for dep in orig_deps - patch_deps:
            removed_edges.append((node, dep))
            logger.debug(f"Edge removed: {node} -> {dep}")
        
        for dep in patch_deps - orig_deps:
            added_edges.append((node, dep))
            logger.debug(f"Edge added: {node} -> {dep}")
    
    # Find remaining cycles
    remaining_cycles = find_cycles_in_graph(patched_graph)
    
    # Check if original cycle nodes still form a cycle
    original_nodes_set = set(cycle_nodes)
    original_cycle_broken = True
    
    for scc in remaining_cycles:
        scc_set = set(scc)
        # Check if any remaining SCC contains multiple original cycle nodes
        overlap = scc_set & original_nodes_set
        if len(overlap) > 1:
            original_cycle_broken = False
            logger.warning(f"Cycle still exists among: {overlap}")
            break
    
    # Build analysis text
    analysis_parts = []
    
    if removed_edges:
        analysis_parts.append(f"Removed {len(removed_edges)} edge(s): " + 
                            ", ".join(f"{a}->{b}" for a, b in removed_edges))
    else:
        analysis_parts.append("No edges were removed from the dependency graph")
    
    if added_edges:
        analysis_parts.append(f"Added {len(added_edges)} new edge(s): " +
                            ", ".join(f"{a}->{b}" for a, b in added_edges))
    
    if remaining_cycles:
        analysis_parts.append(f"Found {len(remaining_cycles)} remaining cycle(s)")
        for i, cycle in enumerate(remaining_cycles[:3]):
            analysis_parts.append(f"  Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
    else:
        analysis_parts.append("No cycles detected in patched code")
    
    if original_cycle_broken:
        analysis_parts.append("✓ The original cycle appears to be broken")
    else:
        analysis_parts.append("✗ The original cycle is NOT broken")
    
    analysis = "\n".join(analysis_parts)
    
    # Calculate confidence
    confidence = 0.8 if removed_edges else 0.3
    if remaining_cycles:
        confidence -= 0.2 * len(remaining_cycles)
    confidence = max(0.1, min(1.0, confidence))
    
    logger.info(f"Cycle verification: broken={original_cycle_broken}, confidence={confidence:.2f}")
    
    return CycleVerificationResult(
        is_broken=original_cycle_broken and not remaining_cycles,
        remaining_cycles=remaining_cycles,
        removed_edges=removed_edges,
        added_edges=added_edges,
        analysis=analysis,
        confidence=confidence
    )
