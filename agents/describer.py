from typing import Dict, Any
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
import json


class DescriberAgent(Agent):
    name = "describer"
    version = "0.2"

    def __init__(self, llm=None, prompt_template: str = None, max_file_chars: int = 4000):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars

    def _build_prompt(self, cycle: Dict[str, Any]) -> str:
        # Try to format the provided template, otherwise create a default prompt
        file_paths = [f.get("path", "") for f in cycle.get("files", [])]
        if self.prompt_template:
            try:
                return self.prompt_template.format(id=cycle.get("id", ""), graph=json.dumps(cycle.get("graph", {})), files=", ".join(file_paths))
            except Exception:
                pass

        base = f"Please describe the cyclic dependency for id={cycle.get('id','')}. Graph: {json.dumps(cycle.get('graph', {}))}. Affected files: {', '.join(file_paths)}."

        # Append truncated file contents to give context to the LLM
        snippets = []
        for f in cycle.get("files", []):
            content = f.get("content", "") or ""
            if len(content) > self.max_file_chars:
                content = content[: self.max_file_chars] + "\n...[truncated]"
            snippets.append(f"--- FILE: {f.get('path','unknown')} ---\n{content}")

        if snippets:
            base += "\n\n" + "\n\n".join(snippets)

        return base

    def run(self, input_data: Dict[str, Any]) -> AgentResult:
        cycle = input_data.get("cycle_spec")
        if cycle is None:
            return AgentResult(status="error", output=None, logs="Missing cycle_spec")

        prompt = self._build_prompt(cycle)

        if self.llm is None:
            # Fallback: simple deterministic description
            nodes = cycle.get("graph", {}).get("nodes", [])
            edges = cycle.get("graph", {}).get("edges", [])
            text = f"Cyclic dependency involving {len(nodes)} node(s): {', '.join(nodes)}. Edges: {edges}"
            description = {"text": text, "highlights": []}
            return AgentResult(status="success", output=description)

        try:
            response = call_llm(self.llm, prompt)
            text = response if isinstance(response, str) else json.dumps(response)
            description = {"text": text, "highlights": []}
            return AgentResult(status="success", output=description)
        except Exception as e:
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
