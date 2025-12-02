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
        for patch in patches:
            path = patch.get("path") or "unknown"
            # normalize path to safe filename
            safe_path = Path("patches") / Path(path)
            full_patch_path = self.artifact_path(artifact_id) / safe_path
            full_patch_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_patch_path, "w", encoding="utf-8") as f:
                f.write(patch.get("patched", ""))

            # diffs
            diff_text = patch.get("diff", "")
            if diff_text:
                diff_path = self.artifact_path(artifact_id) / Path("diffs") / Path(path + ".diff")
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diff_path, "w", encoding="utf-8") as df:
                    df.write(diff_text)

        return p
