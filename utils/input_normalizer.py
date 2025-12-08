from pathlib import Path
from typing import Any, Dict, List, Optional

from models.schemas import CycleSpec, FileSpec, GraphSpec
from utils.logging import get_logger

logger = get_logger("input_normalizer")


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

    # Handle case where raw input is a list (array of cycles without wrapper)
    if isinstance(raw, list):
        logger.info(f"Input is a bare list of {len(raw)} cycle(s)")
        cycles = raw
        if cycle_id:
            cycle = next((c for c in cycles if isinstance(c, dict) and c.get("id") == cycle_id), None)
            if cycle:
                logger.info(f"Selected cycle by ID: {cycle_id}")
            else:
                logger.warning(f"Cycle ID '{cycle_id}' not found, available IDs: {[c.get('id') for c in cycles if isinstance(c, dict)]}")
        if cycle is None and cycles:
            cycle = cycles[0] if isinstance(cycles[0], dict) else None
            if cycle:
                logger.info(f"Using first cycle: {cycle.get('id', 'unknown')}")
    elif isinstance(raw, dict) and "cycles" in raw and isinstance(raw["cycles"], list):
        cycles = raw["cycles"]
        logger.info(f"Input contains {len(cycles)} cycle(s)")
        if cycle_id:
            cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
            if cycle:
                logger.info(f"Selected cycle by ID: {cycle_id}")
            else:
                logger.warning(f"Cycle ID '{cycle_id}' not found, available IDs: {[c.get('id') for c in cycles]}")
        if cycle is None and cycles:
            cycle = cycles[0]
            logger.info(f"Using first cycle: {cycle.get('id', 'unknown')}")
        # Check for project root path
        project_info = raw.get("project", {})
        project_root = project_info.get("root")
        if project_root:
            logger.debug(f"Project root: {project_root}")
    elif isinstance(raw, dict):
        cycle = raw
        logger.debug("Input is a single cycle object")
    else:
        logger.error(f"Unexpected input type: {type(raw)}")
        raise ValueError(f"Unexpected input type: {type(raw)}, expected dict or list")

    if cycle is None:
        logger.error("No cycle found in input JSON")
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
    files_loaded = 0
    files_missing = 0
    files_embedded = 0
    
    for f in files_raw:
        path = f.get("path")
        content = f.get("content")

        if content:
            # Content already embedded in JSON
            files_embedded += 1
            logger.debug(f"Using embedded content for: {path}")
        elif read_files and path:
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
                    files_loaded += 1
                    logger.debug(f"Loaded file: {p} ({len(content)} chars)")
                else:
                    files_missing += 1
                    logger.warning(f"File not found: {path}")
                    content = ""
            except Exception as e:
                files_missing += 1
                logger.warning(f"Failed to read file {path}: {e}")
                content = ""
        else:
            if not read_files:
                logger.debug(f"Skipping file read (--no-read-files): {path}")
            content = ""

        files.append(FileSpec(path=path or "", content=content or ""))
    
    # Summary log for file loading
    logger.info(f"Files: {len(files)} total, {files_loaded} loaded from disk, {files_embedded} embedded, {files_missing} missing")
    if files_missing > 0:
        logger.warning(f"{files_missing} file(s) could not be read - patches may be incomplete")

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
    spec = CycleSpec(
        id=cycle.get("id") or "",
        graph=graph,
        files=files,
        metadata=metadata,
    )
    
    logger.info(f"Normalized cycle '{spec.id}': {len(spec.graph.nodes)} nodes, {len(spec.graph.edges)} edges, {len(spec.files)} files")
    if spec.graph.nodes:
        logger.info(f"Nodes in cycle: {', '.join(spec.graph.nodes[:5])}{'...' if len(spec.graph.nodes) > 5 else ''}")
    
    return spec
