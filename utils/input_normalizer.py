from pathlib import Path
from typing import Any, Dict, Optional


def normalize_input(raw: Dict[str, Any], cycle_id: Optional[str] = None, read_files: bool = True) -> Dict[str, Any]:
    """Normalize various input JSON shapes into the canonical `cycle_spec` shape used by the pipeline.

    canonical shape:
      {
        "id": str,
        "graph": {"nodes": [str], "edges": [...]},
        "files": [{"path": str, "content": str}],
        "metadata": {...}
      }

    - If `raw` contains a top-level `cycles` array, select the cycle by `cycle_id` or first element.
    - For each node, prefer an inlined `content` field; otherwise try to read the file at `path` if `read_files` is True.
    """

    # Choose cycle object
    cycle = None
    if isinstance(raw, dict) and "cycles" in raw and isinstance(raw["cycles"], list):
        cycles = raw["cycles"]
        if cycle_id:
            cycle = next((c for c in cycles if c.get("id") == cycle_id), None)
        if cycle is None and cycles:
            cycle = cycles[0]
    else:
        cycle = raw

    if cycle is None:
        raise ValueError("No cycle found in input JSON")

    nodes = cycle.get("nodes", []) or []
    edges = cycle.get("edges", []) or []

    graph = {"nodes": [n.get("id") for n in nodes], "edges": edges}

    files = []
    for n in nodes:
        path = n.get("path")
        content = n.get("content")

        if not content and read_files and path:
            try:
                p = Path(path)
                if p.is_file():
                    content = p.read_text(encoding="utf-8")
                else:
                    content = ""
            except Exception:
                content = ""

        files.append({"path": path, "content": content or ""})

    metadata = {
        "summary": cycle.get("summary"),
        "impact_score": cycle.get("impact_score"),
        "definitions": cycle.get("definitions"),
        "weighting_explanation": cycle.get("weighting_explanation"),
    }

    canonical = {"id": cycle.get("id"), "graph": graph, "files": files, "metadata": metadata}
    # include original cycle for provenance if needed
    canonical["original_cycle"] = cycle
    return canonical
