import os
import json
from pathlib import Path
from typing import Any, Dict, List


class Persistor:
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = Path(base_dir)

    def _ensure(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, artifact_id: str) -> Path:
        path = self.base_dir / str(artifact_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, artifact_id: str, rel_path: str, obj: Any) -> Path:
        base = self.artifact_path(artifact_id)
        p = base / rel_path
        self._ensure(p)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return p

    def save_text(self, artifact_id: str, rel_path: str, text: str) -> Path:
        base = self.artifact_path(artifact_id)
        p = base / rel_path
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
        base_resolved = base.resolve()

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
                f.write(patch.get("patched", ""))

            # diffs: write alongside in artifacts/{id}/diffs/<same_sanitized_path>.diff
            diff_text = patch.get("diff", "")
            if diff_text:
                diff_rel = Path("diffs") / safe_rel
                diff_path = base.joinpath(diff_rel.with_suffix(diff_rel.suffix + ".diff"))
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
