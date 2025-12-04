from pathlib import Path
from typing import Any, Dict, Optional

from models.schemas import CycleSpec, FileSpec, GraphSpec


def normalize_input(raw: Dict[str, Any], cycle_id: Optional[str] = None, read_files: bool = True) -> CycleSpec:
    """Normalize various input JSON shapes into the canonical `CycleSpec` used by the pipeline.

    Returns a validated CycleSpec Pydantic model.

    Supported input formats:
    1. Legacy format with `nodes` array containing {id, path, content} objects
    2. New format with separate `graph` and `files` arrays

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
        edges = graph_raw.get("edges", [])
        files_raw = cycle.get("files", [])
    elif "nodes" in cycle:
        # Legacy format: nodes array with {id, path, content}
        nodes_raw = cycle.get("nodes", [])
        nodes = [n.get("id") for n in nodes_raw if isinstance(n, dict)]
        edges = cycle.get("edges", [])
        files_raw = [{"path": n.get("path"), "content": n.get("content")} for n in nodes_raw if isinstance(n, dict)]
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
