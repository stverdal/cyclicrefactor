"""Graph cycle detection algorithms.

This module provides algorithms to detect cyclic dependencies in a graph,
using Tarjan's strongly connected components algorithm and cycle enumeration.

Usage:
    from utils.graph_cycle_detector import find_cycles, find_strongly_connected_components
    
    graph = DependencyGraph(...)
    cycles = find_cycles(graph)
    for cycle in cycles:
        print(f"Cycle: {' -> '.join(cycle.nodes)}")
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib

from utils.logging import get_logger

logger = get_logger("graph_cycle_detector")


@dataclass
class CycleDetectorConfig:
    """Configuration for cycle detection."""
    max_cycles: int = 100
    max_cycles_per_scc: int = 50
    include_minor: bool = True
    include_type_only: bool = True


def build_adjacency_list(edges: List[List[str]]) -> Dict[str, Set[str]]:
    """Convert edge list to adjacency list representation.
    
    Args:
        edges: List of [from, to] pairs
        
    Returns:
        Dict mapping each node to set of nodes it points to
    """
    graph: Dict[str, Set[str]] = defaultdict(set)
    malformed_edges = 0
    
    for edge in edges:
        if len(edge) >= 2:
            from_node, to_node = edge[0], edge[1]
            graph[from_node].add(to_node)
            # Ensure to_node exists in graph even if it has no outgoing edges
            if to_node not in graph:
                graph[to_node] = set()
        else:
            malformed_edges += 1
    
    if malformed_edges > 0:
        logger.warning(f"Skipped {malformed_edges} malformed edges (expected [from, to] pairs)")
    
    logger.debug(f"Built adjacency list: {len(graph)} nodes, {sum(len(v) for v in graph.values())} edges")
    
    return dict(graph)


def find_strongly_connected_components(
    adjacency_list: Dict[str, Set[str]]
) -> List[List[str]]:
    """Find all strongly connected components using Tarjan's algorithm.
    
    A strongly connected component (SCC) is a maximal set of vertices
    such that there is a path from each vertex to every other vertex.
    SCCs with more than one node indicate cycles.
    
    This implementation uses an iterative approach with explicit stack
    to avoid Python's recursion limit on large graphs.
    
    Args:
        adjacency_list: Graph as adjacency list
        
    Returns:
        List of SCCs, each SCC is a list of node names
    """
    import time
    start_time = time.time()
    
    logger.debug(f"Starting Tarjan's SCC algorithm on {len(adjacency_list)} nodes")
    
    index_counter = 0
    stack = []
    lowlinks = {}
    index = {}
    on_stack = {}
    sccs = []
    
    # Iterative Tarjan's using explicit call stack
    # Each stack frame: (node, successor_iterator, phase)
    # phase 0 = initial visit, phase 1 = after recursive call
    
    for start_node in adjacency_list:
        if start_node in index:
            continue
            
        call_stack = [(start_node, iter(adjacency_list.get(start_node, set())), 0, None)]
        
        while call_stack:
            node, successors, phase, last_successor = call_stack.pop()
            
            if phase == 0:
                # Initial visit - set up node
                index[node] = index_counter
                lowlinks[node] = index_counter
                index_counter += 1
                stack.append(node)
                on_stack[node] = True
                
                # Push back with phase 1 to continue after processing successors
                call_stack.append((node, successors, 1, None))
                
            elif phase == 1:
                # After recursive call - update lowlink if needed
                if last_successor is not None and last_successor in lowlinks:
                    lowlinks[node] = min(lowlinks[node], lowlinks[last_successor])
                
                # Continue iterating successors
                try:
                    successor = next(successors)
                    
                    if successor not in index:
                        # Push current node back to continue after recursion
                        call_stack.append((node, successors, 1, successor))
                        # Push successor for processing
                        call_stack.append((successor, iter(adjacency_list.get(successor, set())), 0, None))
                    elif on_stack.get(successor, False):
                        # Successor is on stack, update lowlink
                        lowlinks[node] = min(lowlinks[node], index[successor])
                        # Continue with this node
                        call_stack.append((node, successors, 1, None))
                    else:
                        # Continue with this node
                        call_stack.append((node, successors, 1, None))
                        
                except StopIteration:
                    # Done with all successors - check if root of SCC
                    if lowlinks[node] == index[node]:
                        scc = []
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            scc.append(w)
                            if w == node:
                                break
                        sccs.append(scc)
    
    elapsed = time.time() - start_time
    cyclic_sccs = [s for s in sccs if len(s) > 1]
    logger.debug(f"Tarjan's algorithm completed in {elapsed:.3f}s: {len(sccs)} SCCs found, {len(cyclic_sccs)} contain cycles")
    
    return sccs


def find_all_cycles_in_scc(
    scc: List[str],
    adjacency_list: Dict[str, Set[str]],
    max_cycles: int = 100
) -> List[List[str]]:
    """Find all elementary cycles within a strongly connected component.
    
    Uses Johnson's algorithm for finding all elementary cycles.
    
    Args:
        scc: The strongly connected component (list of nodes)
        adjacency_list: Full graph adjacency list
        max_cycles: Maximum number of cycles to find (to prevent explosion)
        
    Returns:
        List of cycles, each cycle is a list of nodes in order
    """
    if len(scc) <= 1:
        return []
    
    logger.debug(f"Finding cycles in SCC with {len(scc)} nodes (max_cycles={max_cycles})")
    
    # Build subgraph for this SCC
    scc_set = set(scc)
    subgraph: Dict[str, Set[str]] = {}
    for node in scc:
        subgraph[node] = adjacency_list.get(node, set()) & scc_set
    
    cycles = []
    blocked = set()
    blocked_map: Dict[str, Set[str]] = defaultdict(set)
    stack = []
    
    def unblock(node):
        blocked.discard(node)
        while blocked_map[node]:
            w = blocked_map[node].pop()
            if w in blocked:
                unblock(w)
    
    def circuit(node, start, path):
        if len(cycles) >= max_cycles:
            return False
        
        found = False
        path.append(node)
        blocked.add(node)
        
        for neighbor in subgraph.get(node, set()):
            if neighbor == start:
                # Found a cycle
                cycles.append(path.copy())
                found = True
            elif neighbor not in blocked:
                if circuit(neighbor, start, path):
                    found = True
        
        if found:
            unblock(node)
        else:
            for neighbor in subgraph.get(node, set()):
                blocked_map[neighbor].add(node)
        
        path.pop()
        return found
    
    # Find cycles starting from each node
    nodes = sorted(scc)  # Sort for deterministic output
    for start in nodes:
        if len(cycles) >= max_cycles:
            break
        circuit(start, start, [])
        # Remove start from consideration in subsequent searches
        for node in subgraph:
            subgraph[node].discard(start)
        blocked.clear()
        blocked_map.clear()
    
    if len(cycles) >= max_cycles:
        logger.warning(f"Hit max_cycles limit ({max_cycles}) in SCC - some cycles may not be reported")
    else:
        logger.debug(f"Found {len(cycles)} cycles in SCC")
    
    return cycles


def generate_cycle_id(nodes: List[str]) -> str:
    """Generate a unique ID for a cycle based on its nodes.
    
    The ID is stable regardless of where in the cycle you start.
    """
    if not nodes:
        return "empty"
    
    # Normalize: start from the lexicographically smallest node
    min_idx = nodes.index(min(nodes))
    normalized = nodes[min_idx:] + nodes[:min_idx]
    
    # Create hash of the normalized cycle
    cycle_str = "->".join(normalized)
    return hashlib.md5(cycle_str.encode()).hexdigest()[:12]


def classify_cycle_severity(
    cycle_nodes: List[str],
    all_cycles: List[List[str]],
    graph_size: int
) -> str:
    """Classify the severity of a cycle.
    
    Args:
        cycle_nodes: Nodes in this cycle
        all_cycles: All detected cycles
        graph_size: Total number of nodes in the graph
        
    Returns:
        "critical", "major", or "minor"
    """
    cycle_size = len(cycle_nodes)
    
    # Large cycles involving many files are critical
    if cycle_size >= 5:
        return "critical"
    
    # Cycles that appear in many other cycles are critical
    node_set = set(cycle_nodes)
    overlap_count = sum(
        1 for other in all_cycles 
        if set(other) != node_set and len(node_set & set(other)) > 0
    )
    if overlap_count >= 3:
        return "critical"
    
    # Direct 2-node cycles are usually major
    if cycle_size == 2:
        return "major"
    
    # Small isolated cycles are minor
    if cycle_size <= 3 and overlap_count == 0:
        return "minor"
    
    return "major"


def find_cycles(
    dependency_graph: "DependencyGraph",
    config: Optional[CycleDetectorConfig] = None,
    max_cycles_per_scc: int = 50,
    include_type_only: bool = True
) -> List["DetectedCycle"]:
    """Find all cyclic dependencies in a dependency graph.
    
    Args:
        dependency_graph: The dependency graph to analyze
        config: Optional configuration object
        max_cycles_per_scc: Max cycles to find per SCC (prevents explosion)
        include_type_only: Include type-only import cycles
        
    Returns:
        List of DetectedCycle objects
    """
    from models.schemas import DetectedCycle
    
    # Use config if provided
    if config:
        max_cycles_per_scc = config.max_cycles_per_scc
        include_type_only = config.include_type_only
        logger.debug(f"Cycle detection config: max_cycles_per_scc={max_cycles_per_scc}, include_type_only={include_type_only}")
    
    edges = dependency_graph.edges
    original_edge_count = len(edges)
    
    # Filter edges if needed
    if not include_type_only:
        type_only_imports = {
            (i.source_file, i.imported_from)
            for i in dependency_graph.imports
            if i.import_type == "type-only"
        }
        edges = [e for e in edges if tuple(e) not in type_only_imports]
        if len(edges) < original_edge_count:
            logger.info(f"Filtered out {original_edge_count - len(edges)} type-only import edges")
    
    if len(dependency_graph.imports) == 0:
        logger.debug("No import metadata available - type-only filtering will have no effect")
    
    # Build adjacency list
    adj_list = build_adjacency_list(edges)
    
    logger.info(f"Analyzing graph with {len(adj_list)} nodes")
    
    # Find strongly connected components
    sccs = find_strongly_connected_components(adj_list)
    
    # Filter to SCCs with more than one node (actual cycles)
    cyclic_sccs = [scc for scc in sccs if len(scc) > 1]
    
    logger.info(f"Found {len(cyclic_sccs)} strongly connected components with cycles")
    
    if not cyclic_sccs:
        return []
    
    # Find individual cycles within each SCC
    all_cycles = []
    for scc in cyclic_sccs:
        cycles = find_all_cycles_in_scc(scc, adj_list, max_cycles_per_scc)
        all_cycles.extend(cycles)
    
    logger.info(f"Found {len(all_cycles)} elementary cycles")
    
    # Convert to DetectedCycle objects
    detected_cycles = []
    graph_size = len(adj_list)
    severity_counts = {"critical": 0, "major": 0, "minor": 0}
    
    logger.debug(f"Converting {len(all_cycles)} raw cycles to DetectedCycle objects")
    
    for cycle_nodes in all_cycles:
        # Get edges that form this cycle
        cycle_set = set(cycle_nodes)
        cycle_edges = []
        for i, node in enumerate(cycle_nodes):
            next_node = cycle_nodes[(i + 1) % len(cycle_nodes)]
            cycle_edges.append([node, next_node])
        
        cycle_id = generate_cycle_id(cycle_nodes)
        severity = classify_cycle_severity(cycle_nodes, all_cycles, graph_size)
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Create description
        if len(cycle_nodes) == 2:
            desc = f"Direct circular dependency between {cycle_nodes[0]} and {cycle_nodes[1]}"
        else:
            desc = f"Circular dependency chain: {' → '.join(cycle_nodes)} → {cycle_nodes[0]}"
        
        detected_cycles.append(DetectedCycle(
            id=f"cycle_{cycle_id}",
            nodes=cycle_nodes,
            edges=cycle_edges,
            severity=severity,
            cycle_type="direct" if len(cycle_nodes) == 2 else "indirect",
            description=desc,
        ))
    
    # Sort by severity (critical first) then by size
    severity_order = {"critical": 0, "major": 1, "minor": 2}
    detected_cycles.sort(key=lambda c: (severity_order.get(c.severity, 1), -len(c.nodes)))
    
    logger.info(f"Cycle detection complete: {len(detected_cycles)} cycles "
                f"(critical={severity_counts['critical']}, major={severity_counts['major']}, minor={severity_counts['minor']})")
    
    return detected_cycles


def deduplicate_cycles(cycles: List["DetectedCycle"]) -> List["DetectedCycle"]:
    """Remove duplicate cycles (same nodes, different starting points).
    
    Args:
        cycles: List of detected cycles
        
    Returns:
        Deduplicated list
    """
    seen_ids = set()
    unique = []
    
    for cycle in cycles:
        if cycle.id not in seen_ids:
            seen_ids.add(cycle.id)
            unique.append(cycle)
    
    return unique


def get_cycle_summary(cycles: List["DetectedCycle"]) -> Dict[str, int]:
    """Get a summary of detected cycles.
    
    Returns:
        Dict with counts by severity
    """
    summary = {
        "total": len(cycles),
        "critical": 0,
        "major": 0,
        "minor": 0,
        "direct": 0,
        "indirect": 0,
    }
    
    for cycle in cycles:
        summary[cycle.severity] = summary.get(cycle.severity, 0) + 1
        summary[cycle.cycle_type] = summary.get(cycle.cycle_type, 0) + 1
    
    return summary
