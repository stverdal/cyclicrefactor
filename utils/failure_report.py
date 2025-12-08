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
    description_text: Optional[str] = None,
    last_proposal: Optional[Dict[str, Any]] = None,
    last_validation: Optional[Dict[str, Any]] = None,
    validation_history: Optional[List[Dict[str, Any]]] = None,
    attempt_summaries: Optional[List[str]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
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

    # Read snippets (respect config knobs if provided)
    max_log_chars = 8000
    max_llm_excerpt = 2000
    if config and isinstance(config, dict):
        rep_cfg = config.get("report", {}) if isinstance(config, dict) else {}
        max_log_chars = int(rep_cfg.get("max_log_chars", max_log_chars))
        max_llm_excerpt = int(rep_cfg.get("max_llm_excerpt", max_llm_excerpt))

    llm_text = _read_file_snippet(llm_log, max_chars=max_log_chars)
    main_text = _read_file_snippet(main_log, max_chars=max_log_chars)

    # Extract recent llm calls/responses
    llm_calls = _extract_recent_llm_calls(llm_text, limit=8)
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

    # Proposal / patch summary
    patch_count = 0
    new_files_count = 0
    failed_patches = []
    sample_failed = []
    if last_proposal:
        patches = last_proposal.get("patches") if isinstance(last_proposal, dict) else None
        if patches is None and hasattr(last_proposal, 'patches'):
            patches = getattr(last_proposal, 'patches')
        patch_count = len(patches) if patches else 0
        for p in (patches or []):
            status = p.get("status") if isinstance(p, dict) else getattr(p, 'status', None)
            path = p.get("path") if isinstance(p, dict) else getattr(p, 'path', None)
            if status == "new_file" or p.get("is_new_file"):
                new_files_count += 1
            if status in ("failed", "reverted") or p.get("warnings"):
                failed_patches.append({"path": path, "status": status, "warnings": p.get("warnings", [])})
        report["patch_summary"] = {
            "total": patch_count,
            "new_files": new_files_count,
            "failed_count": len(failed_patches),
        }
        report["timeline"].append({"ts": None, "event": f"Proposal generated: {patch_count} patch(es), {new_files_count} new file(s)", "details": []})

    # Validation summary
    issues_count = 0
    if last_validation:
        issues = last_validation.get("issues") if isinstance(last_validation, dict) else None
        issues_count = len(issues) if issues else 0
        report["timeline"].append({"ts": None, "event": f"Validator reported: {issues_count} issue(s)", "details": []})
        # top validation issues
        for i, iss in enumerate((issues or [])[:5]):
            if isinstance(iss, dict):
                report["top_validation_issues"].append({
                    "path": iss.get("path"),
                    "line": iss.get("line"),
                    "comment": iss.get("comment"),
                    "issue_type": iss.get("issue_type"),
                })

    # Build a structured human-readable narrative
    # Sections: Summary, What we tried, What happened (evidence), Probable causes, Next steps
    summary_parts = []
    if issues_count > 0:
        summary_parts.append(f"Refactor attempt failed: validator reported {issues_count} issue(s)")
    elif failed_patches:
        summary_parts.append(f"Refactor attempt partially failed: {len(failed_patches)} patch(es) had issues")
    else:
        summary_parts.append("Refactor attempt did not succeed")

    what_we_tried = []
    if description_text:
        what_we_tried.append(f"Description: {description_text[:300].strip()}")
    if attempt_summaries:
        what_we_tried.append("Attempts: " + " | ".join(attempt_summaries))
    else:
        what_we_tried.append(f"Generated proposal with {patch_count} patches")

    evidence_lines = []
    if llm_calls:
        for c in llm_calls[:3]:
            # extract prompt/response excerpts
            txt = c.get("text", "")
            prompt_block = ""
            response_block = ""
            m_prompt = re.search(r"PROMPT\s*-+\n([\s\S]{0,%d})" % max_llm_excerpt, txt)
            if m_prompt:
                prompt_block = m_prompt.group(1).strip()
            m_resp = re.search(r"RESPONSE\s*-+\n([\s\S]{0,%d})" % max_llm_excerpt, txt)
            if m_resp:
                response_block = m_resp.group(1).strip()
            header = f"LLM {c.get('kind')} {c.get('agent')} {c.get('stage')} ID={c.get('id')}"
            evidence_lines.append(header)
            if prompt_block:
                evidence_lines.append("Prompt excerpt: " + prompt_block.replace('\n', ' ')[:500])
            if response_block:
                evidence_lines.append("Response excerpt: " + response_block.replace('\n', ' ')[:500])

    # include errors and top validation issues as evidence
    if errors:
        evidence_lines.append("Recent errors: ")
        evidence_lines.extend(errors[:5])
    for tv in report.get("top_validation_issues", [])[:3]:
        evidence_lines.append(f"Validation issue: {tv.get('path')}:{tv.get('line')} - {tv.get('comment')}")

    probable_causes = []
    if issues_count > 0:
        probable_causes.append("Validation failures indicate mismatches between proposed changes and code expectations (types/signatures).")
    if failed_patches:
        probable_causes.append("Patch application failures likely due to incorrect line numbers or context drift.")
    if not probable_causes:
        probable_causes.append("Insufficient context or an ambiguous refactoring strategy.")

    next_steps = []
    next_steps.append("Inspect the 'log_report.log_snippets' and 'top_validation_issues' for precise failures.")
    next_steps.append("If an interface was created, verify its method signatures match consumers.")
    next_steps.append("Consider enabling 'scaffolding_mode' or increasing context for the LLM and retry.")

    narrative = {
        "summary": " ".join(summary_parts),
        "what_we_tried": what_we_tried,
        "evidence": evidence_lines,
        "probable_causes": probable_causes,
        "next_steps": next_steps,
    }
    report["narrative"] = narrative

    # Also include recommended actions (mirrors narrative next_steps)
    report["recommended_actions"] = next_steps

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
            # include persisted proposal/validation artifacts if present (short excerpts)
            prop_path = os.path.join(artifact_path, "refactor_proposal.json")
            if os.path.exists(prop_path):
                try:
                    with open(prop_path, "r", encoding="utf-8") as pf:
                        prop = json.load(pf)
                        report["persisted_proposal_excerpt"] = {
                            "patches_count": len(prop.get("patches", [])),
                            "rationale": (prop.get("rationale", "")[:400])
                        }
                except Exception:
                    pass
            val_path = os.path.join(artifact_path, "validation/report.json")
            if os.path.exists(val_path):
                try:
                    with open(val_path, "r", encoding="utf-8") as vf:
                        val = json.load(vf)
                        report["persisted_validation_excerpt"] = {
                            "approved": val.get("approved"),
                            "issues_count": len(val.get("issues", []))
                        }
                except Exception:
                    pass
    except Exception:
        pass

    # Summary is a short sentence for report heading
    report["summary"] = (narrative[:400])

    return report
