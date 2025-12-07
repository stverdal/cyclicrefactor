import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger("persistence")


class Persistor:
    def __init__(self, base_dir: str = "artifacts", dry_run: bool = False, log_writes: bool = True):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.log_writes = log_writes
        self._write_log: List[Dict[str, Any]] = []  # Track what would be written in dry-run

    def _ensure(self, path: Path):
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _log_write(self, path: Path, content_type: str, size: int):
        """Log a write operation (for dry-run tracking)."""
        entry = {
            "path": str(path),
            "type": content_type,
            "size": size,
        }
        self._write_log.append(entry)
        if self.log_writes:
            logger.info(f"[DRY-RUN] Would write {content_type} ({size} bytes) to: {path}")

    def get_write_log(self) -> List[Dict[str, Any]]:
        """Get list of all writes that would have been made."""
        return self._write_log.copy()

    def artifact_path(self, artifact_id: str) -> Path:
        path = self.base_dir / str(artifact_id)
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, artifact_id: str, rel_path: str, obj: Any) -> Path:
        base = self.artifact_path(artifact_id)
        p = base / rel_path
        
        if self.dry_run:
            content = json.dumps(obj, indent=2, ensure_ascii=False)
            self._log_write(p, "json", len(content))
        else:
            self._ensure(p)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        return p

    def save_text(self, artifact_id: str, rel_path: str, text: str) -> Path:
        base = self.artifact_path(artifact_id)
        p = base / rel_path
        
        if self.dry_run:
            self._log_write(p, "text", len(text))
        else:
            self._ensure(p)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        return p

    def persist_cycle_input(self, artifact_id: str, cycle_spec: Dict[str, Any]) -> Path:
        return self.save_json(artifact_id, "input/cycle_spec.json", cycle_spec)

    def persist_description(self, artifact_id: str, description_text: str) -> Path:
        return self.save_text(artifact_id, "describer/description.txt", description_text)

    def persist_llm_response(self, artifact_id: str, agent_name: str, response: Any) -> Path:
        # store the raw response as json if possible, else as text
        try:
            return self.save_json(artifact_id, f"llm/{agent_name}_response.json", response)
        except Exception:
            return self.save_text(artifact_id, f"llm/{agent_name}_response.txt", str(response))

    def persist_proposal(self, artifact_id: str, proposal: Dict[str, Any]) -> Path:
        p = self.save_json(artifact_id, "proposal/proposal.json", proposal)
        # persist patches as individual files and diffs
        patches = proposal.get("patches", [])
        base = self.artifact_path(artifact_id)
        
        if not self.dry_run:
            base_resolved = base.resolve()
        else:
            base_resolved = base  # Skip resolve in dry-run

        for patch in patches:
            orig_path = patch.get("path") or "unknown"
            # Build a sanitized relative path under artifacts/{id}/patches
            ppath = Path(orig_path)

            if ppath.is_absolute():
                # drop the drive/anchor and use remaining parts
                parts = list(ppath.parts)[1:]
                if not parts:
                    parts = [ppath.name]
            else:
                # remove any parent traversal components for safety
                parts = [part for part in ppath.parts if part not in ("..", ".")]

            safe_rel = Path("patches").joinpath(*parts)
            full_patch_path = base.joinpath(safe_rel)
            
            patched_content = patch.get("patched", "")
            
            if self.dry_run:
                self._log_write(full_patch_path, "patch", len(patched_content))
            else:
                # Ensure parent exists
                full_patch_path.parent.mkdir(parents=True, exist_ok=True)

                # Defensive check: ensure we are writing inside the artifact dir
                try:
                    full_patch_path_resolved = full_patch_path.resolve()
                    # This will raise ValueError if not a subpath
                    full_patch_path_resolved.relative_to(base_resolved)
                except Exception:
                    # Fallback to a safe location: artifacts/{id}/patches/<basename>
                    full_patch_path = base / "patches" / Path(ppath.name)
                    full_patch_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_patch_path, "w", encoding="utf-8") as f:
                    f.write(patched_content)

            # diffs: write alongside in artifacts/{id}/diffs/<same_sanitized_path>.diff
            diff_text = patch.get("diff", "")
            if diff_text:
                diff_rel = Path("diffs") / safe_rel
                diff_path = base.joinpath(diff_rel.with_suffix(diff_rel.suffix + ".diff"))
                
                if self.dry_run:
                    self._log_write(diff_path, "diff", len(diff_text))
                else:
                    try:
                        diff_path.parent.mkdir(parents=True, exist_ok=True)
                        diff_path_resolved = diff_path.resolve()
                        diff_path_resolved.relative_to(base_resolved)
                    except Exception:
                        diff_path = base / "diffs" / Path(ppath.name + ".diff")
                        diff_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(diff_path, "w", encoding="utf-8") as df:
                        df.write(diff_text)

        return p
    
    def persist_failed_patches(
        self, 
        artifact_id: str, 
        failed_patches: List[Dict[str, Any]]
    ) -> Optional[Path]:
        """Persist detailed information about failed patches for later analysis.
        
        Args:
            artifact_id: The artifact/run identifier
            failed_patches: List of dicts containing:
                - path: File path that failed
                - reason: Why the patch failed
                - original_preview: First/last lines of original file
                - llm_patch: What the LLM tried to do
                - warnings: List of warning messages
                - search_replace_ops: The individual S/R operations if applicable
                
        Returns:
            Path to the saved JSON file, or None if no failures
        """
        if not failed_patches:
            return None
        
        # Create a structured failure report
        failure_report = {
            "artifact_id": artifact_id,
            "total_failures": len(failed_patches),
            "failures": failed_patches,
        }
        
        return self.save_json(artifact_id, "failures/failed_patches.json", failure_report)
