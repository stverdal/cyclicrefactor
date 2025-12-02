from typing import Dict, Any, List
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
import json
import difflib
import re


class RefactorAgent(Agent):
    name = "refactor"
    version = "0.3"

    def __init__(self, llm=None, prompt_template: str = None, max_file_chars: int = 4000):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars

    def _build_prompt(self, cycle: Dict[str, Any], description: Dict[str, Any]) -> str:
        file_paths = [f.get("path") for f in cycle.get("files", [])]
        if self.prompt_template:
            try:
                return self.prompt_template.format(id=cycle.get("id", ""), graph=json.dumps(cycle.get("graph", {})), files=", ".join(file_paths), description=description.get("text", ""))
            except Exception:
                pass

        base = f"You are a refactoring assistant. The cyclic dependency: {description.get('text','')}.\nFiles: {', '.join(file_paths)}.\nPlease propose refactorings and return patched file contents."

        snippets = []
        for f in cycle.get("files", []):
            content = f.get("content", "") or ""
            if len(content) > self.max_file_chars:
                content = content[: self.max_file_chars] + "\n...[truncated]"
            snippets.append(f"--- FILE: {f.get('path','unknown')} ---\n{content}")

        if snippets:
            base += "\n\n" + "\n\n".join(snippets)

        base += "\n\nReturn results either as JSON: {\"patches\": [{\"path\":..., \"patched\": ...}], \"notes\":...} or as plain text with markers '--- FILE: <path> ---' followed by patched content."

        return base

    def _parse_json_patches(self, text: str) -> List[Dict[str, str]]:
        try:
            data = json.loads(text)
            patches = []
            for p in data.get("patches", []):
                patches.append({
                    "path": p.get("path"),
                    "patched": p.get("patched")
                })
            return patches
        except Exception:
            return []

    def _parse_marker_patches(self, text: str) -> List[Dict[str, str]]:
        # Split by marker lines like '--- FILE: path ---'
        pattern = r"--- FILE: (.+?) ---\n"
        parts = re.split(pattern, text)
        # re.split will produce [pre, path1, content1, path2, content2, ...]
        patches = []
        if len(parts) < 3:
            return patches
        # drop leading preamble
        it = iter(parts[1:])
        for path, content in zip(it, it):
            patches.append({"path": path.strip(), "patched": content.strip()})
        return patches

    def _infer_patches(self, llm_response: Any, cycle_files: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # Try structured JSON first
        text = llm_response if isinstance(llm_response, str) else json.dumps(llm_response)

        patches = self._parse_json_patches(text)
        if patches:
            return patches

        patches = self._parse_marker_patches(text)
        if patches:
            return patches

        # Nothing parsed: return empty => no-op
        return []

    def _make_unified_diff(self, original: str, patched: str, path: str) -> str:
        orig_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)
        diff = difflib.unified_diff(orig_lines, patched_lines, fromfile=f"a/{path}", tofile=f"b/{path}")
        return "".join(diff)

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        cycle = input_data.get("cycle_spec")
        description = input_data.get("cycle_description")
        if cycle is None or description is None:
            return AgentResult(status="error", output=None, logs="Missing inputs")

        prompt = self._build_prompt(cycle, description)

        # If no LLM available just return original files as no-op patches
        if self.llm is None:
            patches = [
                {"path": f.get("path"), "original": f.get("content"), "patched": f.get("content"), "diff": ""} for f in cycle.get("files", [])
            ]
            proposal = {"patches": patches, "rationale": "No-op (no LLM provided)", "llm_response": None}
            return AgentResult(status="success", output=proposal)

        try:
            llm_response = call_llm(self.llm, prompt)

            inferred = self._infer_patches(llm_response, cycle.get("files", []))

            # Build final patches list merging originals with patched content (if present)
            patches_out = []
            for f in cycle.get("files", []):
                path = f.get("path")
                original = f.get("content") or ""
                patched_entry = next((p for p in inferred if p.get("path") == path), None)
                if patched_entry is None:
                    # try basename match
                    basename = path.split("/")[-1]
                    patched_entry = next((p for p in inferred if p.get("path") == basename), None)

                patched = patched_entry.get("patched") if patched_entry else original

                diff = self._make_unified_diff(original, patched, path) if patched != original else ""

                patches_out.append({"path": path, "original": original, "patched": patched, "diff": diff})

            proposal = {"patches": patches_out, "rationale": "LLM produced proposal", "llm_response": llm_response}
            return AgentResult(status="success", output=proposal)
        except Exception as e:
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
