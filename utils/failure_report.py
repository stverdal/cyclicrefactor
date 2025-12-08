import os
import re
import json
from typing import Any, Dict, List, Optional


def _read_file_snippet(path: str, max_chars: int = 2000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            if len(data) > max_chars:
                return data[-max_chars:]
            return data
    except Exception:
        return ""


def _split_llm_blocks(text: str) -> List[str]:
    # llm_io logger uses a separator of '=' * 80
    parts = [p.strip() for p in re.split(r"\n={5,}\n", text) if p.strip()]
    return parts


def _extract_recent_llm_calls(llm_text: str, limit: int = 3) -> List[Dict[str, Any]]:
    blocks = _split_llm_blocks(llm_text)
    calls = []
    for b in reversed(blocks):
        header_match = re.search(r"LLM (CALL|RESPONSE) \| Agent: (?P<agent>[^|]+) \| Stage: (?P<stage>[^|]+) \| ID: (?P<id>\S+)", b)
        if header_match:
            kind = header_match.group(1)
            agent = header_match.group("agent").strip()
            stage = header_match.group("stage").strip()
            call_id = header_match.group("id").strip()
            calls.append({"kind": kind, "agent": agent, "stage": stage, "id": call_id, "text": b})
        if len(calls) >= limit:
            break
    return list(reversed(calls))


def _extract_errors(log_text: str, limit: int = 10) -> List[str]:
    lines = [l for l in log_text.splitlines() if ("ERROR" in l or "Traceback" in l or "Exception" in l)]
    return lines[-limit:]


def build_failure_report(
    artifact_dir: str,
    artifact_id: str,
    cycle_id: Optional[str] = None,
    last_proposal: Optional[Dict[str, Any]] = None,
    last_validation: Optional[Dict[str, Any]] = None,
    validation_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a human-readable failure report by extracting recent logs and creating
    a concise narrative. This is intentionally conservative: it only reports
    facts extracted from logs and inputs.
    """
    report = {
        "artifact_id": artifact_id,
        "cycle_id": cycle_id,
        "summary": "",
        "timeline": [],
        "llm_calls": [],
        "log_snippets": [],
        "errors": [],
        "narrative": "",
        "recommended_actions": [],
    }

    # Determine log file paths from config or defaults
    main_log = None
    llm_log = None
    if config and isinstance(config, dict):
        main_log = config.get("logging", {}).get("log_file") or config.get("logging", {}).get("log_file", "langCodeUnderstanding.log")
        llm_log = config.get("logging", {}).get("llm_io_log_file") or config.get("logging", {}).get("llm_io_log_file", "llm_io.log")
    else:
        main_log = "langCodeUnderstanding.log"
        llm_log = "llm_io.log"

    # Read snippets
    llm_text = _read_file_snippet(llm_log, max_chars=8000)
    main_text = _read_file_snippet(main_log, max_chars=8000)

    # Extract recent llm calls/responses
    llm_calls = _extract_recent_llm_calls(llm_text, limit=5)
    report["llm_calls"] = llm_calls

    # Extract errors from main log
    errors = _extract_errors(main_text, limit=10)
    report["errors"] = errors

    # Add log snippets that mention the cycle_id or artifact_id if present
    if cycle_id:
        pattern = re.compile(re.escape(cycle_id))
        matches = [l for l in (main_text + "\n" + llm_text).splitlines() if pattern.search(l)]
        if matches:
            report["log_snippets"].extend(matches[-20:])

    # If no cycle-specific snippets, include the last few error lines and LLM blocks
    if not report["log_snippets"]:
        if errors:
            report["log_snippets"].extend(errors[:5])
        # add short llm call excerpts (headers)
        for c in llm_calls:
            header = f"{c.get('kind')} {c.get('agent')} {c.get('stage')} ID={c.get('id')}"
            report["log_snippets"].append(header)

    # Basic timeline events (derived from provided inputs, conservative)
    if llm_calls:
        report["timeline"].append({
            "ts": None,
            "event": f"Recent LLM activity: {len(llm_calls)} call(s)",
            "details": [c.get("id") for c in llm_calls],
        })

    if last_proposal:
        patches = last_proposal.get("patches") if isinstance(last_proposal, dict) else None
        if patches is None and hasattr(last_proposal, 'patches'):
            patches = getattr(last_proposal, 'patches')
        patch_count = len(patches) if patches else 0
        report["timeline"].append({"ts": None, "event": f"Proposal generated: {patch_count} patch(es)", "details": []})

    if last_validation:
        issues = last_validation.get("issues") if isinstance(last_validation, dict) else None
        issues_count = len(issues) if issues else 0
        report["timeline"].append({"ts": None, "event": f"Validator reported: {issues_count} issue(s)", "details": []})

    # Build a short human-readable narrative from events
    parts = []
    if last_proposal:
        parts.append(f"LLM produced a proposal with {patch_count} patch(es).")
    if last_validation:
        parts.append(f"Validator returned {issues_count} issue(s).")
    if errors:
        parts.append(f"Log contains {len(errors)} error/exception entries; see snippets for details.")
    if not parts:
        parts.append("A refactor attempt did not complete; logs contain no clear structured events.")

    narrative = " ".join(parts)
    report["narrative"] = narrative

    # Recommended actions (conservative, based on observed items)
    recs = []
    if errors:
        recs.append("Inspect the error traces in the logs to identify root exceptions.")
    if last_validation and issues_count > 0:
        recs.append("Review validator issues and address the top 1-3 failures (type/signature mismatches, missing methods).")
    if last_proposal and patch_count == 0:
        recs.append("LLM proposal contained no patches; consider increasing context or changing strategy to scaffolding.")
    if not recs:
        recs.append("Review the persisted artifacts in the run folder for more context.")

    report["recommended_actions"] = recs

    # Also include raw snippets for artifacts folder
    try:
        artifact_path = os.path.join(artifact_dir, artifact_id)
        if os.path.isdir(artifact_path):
            # include run metadata if present
            meta_path = os.path.join(artifact_path, "run_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        report["run_metadata"] = json.load(mf)
                except Exception:
                    pass
    except Exception:
        pass

    # Summary is a short sentence for report heading
    report["summary"] = (narrative[:400])

    return report
