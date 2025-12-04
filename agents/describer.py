from typing import Dict, Any, Optional, Union
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet
from utils.prompt_loader import load_template, safe_format
from models.schemas import CycleSpec, CycleDescription
import json


class DescriberAgent(Agent):
    name = "describer"
    version = "0.4"

    def __init__(self, llm=None, prompt_template: str = None, max_file_chars: int = 4000):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars

    def _build_prompt(self, cycle: CycleSpec) -> str:
        # Try to format the provided template, otherwise create a default prompt
        file_paths = cycle.get_file_paths()

        # Prepare file content snippets by selecting relevant regions (imports,
        # defs, and symbol mentions) rather than the file head so prompts stay
        # concise and focused. Each snippet is truncated to `max_file_chars`.
        snippets = []
        # Convert CycleSpec to dict for snippet selector compatibility
        cycle_dict = cycle.model_dump()
        for f in cycle.files:
            content = f.content or ""
            snippet = select_relevant_snippet(content, f.path, cycle_dict, self.max_file_chars)
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")

        file_snippets = "\n\n".join(snippets) if snippets else ""

        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                file_snippets=file_snippets,
            )
            # Append snippets if template didn't already include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            return result

        base = f"Please describe the cyclic dependency for id={cycle.id}. Graph: {json.dumps(cycle.graph.model_dump())}. Affected files: {', '.join(file_paths)}."

        if file_snippets:
            base += "\n\n" + file_snippets

        return base

    def run(self, cycle_spec: Union[CycleSpec, Dict[str, Any]], prompt: str = None) -> AgentResult:
        """Describe the cyclic dependency.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with CycleDescription output.
        """
        # Convert dict to CycleSpec if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)

        prompt_text = self._build_prompt(cycle_spec)

        if self.llm is None:
            # Fallback: simple deterministic description
            nodes = cycle_spec.graph.nodes
            edges = cycle_spec.graph.edges
            text = f"Cyclic dependency involving {len(nodes)} node(s): {', '.join(nodes)}. Edges: {edges}"
            description = CycleDescription(text=text, highlights=[])
            return AgentResult(status="success", output=description.model_dump())

        try:
            response = call_llm(self.llm, prompt_text)
            text = response if isinstance(response, str) else json.dumps(response)
            description = CycleDescription(text=text, highlights=[])
            return AgentResult(status="success", output=description.model_dump())
        except Exception as e:
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
