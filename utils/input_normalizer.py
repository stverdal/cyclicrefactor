from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import CycleSpec, FileSpec, GraphSpec


def _normalize_edges(edges_raw: List[Any]) -> List[List[str]]:
    """Convert various edge formats to canonical [source, target] lists.
    
    Supports:
    - [source, target] lists (pass through)
    - {"source": ..., "target": ...} objects (convert to list)
    """
    edges = []
    for e in edges_raw:
        if isinstance(e, list) and len(e) >= 2:
            # Already in [source, target] format
            edges.append([str(e[0]), str(e[1])])
        elif isinstance(e, dict) and "source" in e and "target" in e:
            # Object format: {source, target, relation?}
            edges.append([str(e["source"]), str(e["target"])])
        # Skip invalid entries
    return edges


def normalize_input(raw: Dict[str, Any], cycle_id: Optional[str] = None, read_files: bool = True) -> CycleSpec:
    """Normalize various input JSON shapes into the canonical `CycleSpec` used by the pipeline.

    Returns a validated CycleSpec Pydantic model.

    Supported input formats:
    1. Legacy format with `nodes` array containing {id, path, content} objects
    2. New format with separate `graph` and `files` arrays
    3. Cycle detector format with nodes as {id, type, name, path} and edges as {source, target, relation}

    - If `raw` contains a top-level `cycles` array, select the cycle by `cycle_id` or first element.
    - For each file, prefer an inlined `content` field; otherwise try to read the file at `path` if `read_files` is True.
    """

    # Choose cycle object
    cycle = None
    project_root = None

    if isinstance(raw, dict) and "cycles" in raw and isinstance(raw["cycles"], list):
        cycles = raw["cycles"]
        if cycle_id:
            cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
        if cycle is None and cycles:
            cycle = cycles[0]
        # Check for project root path
        project_info = raw.get("project", {})
        project_root = project_info.get("root")
    else:
        cycle = raw

    if cycle is None:
        raise ValueError("No cycle found in input JSON")

    # Detect format and normalize
    if "graph" in cycle and "files" in cycle:
        # New format: graph and files are separate
        graph_raw = cycle.get("graph", {})
        nodes = graph_raw.get("nodes", [])
        edges_raw = graph_raw.get("edges", [])
        edges = _normalize_edges(edges_raw)
        files_raw = cycle.get("files", [])
    elif "nodes" in cycle:
        # Legacy/cycle-detector format: nodes array with {id, path, ...} or {id, type, name, path}
        nodes_raw = cycle.get("nodes", [])
        nodes = [n.get("id") or n.get("name") for n in nodes_raw if isinstance(n, dict)]
        edges_raw = cycle.get("edges", [])
        edges = _normalize_edges(edges_raw)
        # Extract file info from nodes
        files_raw = [{"path": n.get("path"), "content": n.get("content")} for n in nodes_raw if isinstance(n, dict) and n.get("path")]
    else:
        # Fallback: assume it's already canonical
        nodes = []
        edges = []
        files_raw = []

    # Build graph
    graph = GraphSpec(nodes=nodes, edges=edges)

    # Load file contents if needed
    files = []
    for f in files_raw:
        path = f.get("path")
        content = f.get("content")

        if (not content or content is None) and read_files and path:
            try:
                # Try path as-is first
                p = Path(path)
                if not p.is_file() and project_root:
                    # Try relative to project root
                    p = Path(project_root) / Path(path).name
                if not p.is_file():
                    # Try just the path from current working directory
                    p = Path(path)

                if p.is_file():
                    content = p.read_text(encoding="utf-8")
                else:
                    content = ""
            except Exception:
                content = ""

        files.append(FileSpec(path=path or "", content=content or ""))

    # Build metadata
    metadata = cycle.get("metadata", {})
    if not metadata:
        metadata = {
            "summary": cycle.get("summary"),
            "impact_score": cycle.get("impact_score"),
            "definitions": cycle.get("definitions"),
            "weighting_explanation": cycle.get("weighting_explanation"),
        }

    # Return validated CycleSpec model
    return CycleSpec(
        id=cycle.get("id") or "",
        graph=graph,
        files=files,
        metadata=metadata,
    )
