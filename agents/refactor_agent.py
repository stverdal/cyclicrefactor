from typing import Dict, Any, List, Optional, Union
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.snippet_selector import select_relevant_snippet
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from models.schemas import CycleSpec, CycleDescription, RefactorProposal, Patch, ValidationReport
import json
import difflib
import re

logger = get_logger("refactor")


class RefactorAgent(Agent):
    name = "refactor"
    version = "0.6"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        max_file_chars: int = 4000,
        rag_service=None
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_file_chars = max_file_chars
        self.rag_service = rag_service
    
    def _get_rag_context(self, cycle: CycleSpec) -> str:
        """Retrieve relevant context from RAG for refactoring guidance."""
        if self.rag_service is None:
            return ""
        
        try:
            # Query for refactoring patterns and best practices
            nodes = ", ".join(cycle.graph.nodes[:5])
            query = f"dependency inversion refactoring patterns {nodes}"
            
            context = self.rag_service.get_relevant_context(
                query,
                k=3,
                max_length=2000
            )
            
            if context:
                logger.debug(f"Retrieved RAG context for refactoring: {len(context)} chars")
                return context
        except Exception as e:
            logger.warning(f"Failed to retrieve RAG context: {e}")
        
        return ""

    def _build_prompt(self, cycle: CycleSpec, description: CycleDescription) -> str:
        file_paths = cycle.get_file_paths()
        if self.prompt_template:
            try:
                return self.prompt_template.format(
                    id=cycle.id,
                    graph=json.dumps(cycle.graph.model_dump()),
                    files=", ".join(file_paths),
                    description=description.text,
                )
            except Exception:
                pass

        base = f"You are a refactoring assistant. The cyclic dependency: {description.text}.\nFiles: {', '.join(file_paths)}.\nPlease propose refactorings and return patched file contents."

        snippets = []
        for f in cycle.files:
            content = f.content or ""
            if len(content) > self.max_file_chars:
                content = content[: self.max_file_chars] + "\n...[truncated]"
            snippets.append(f"--- FILE: {f.path} ---\n{content}")

        if snippets:
            base += "\n\n" + "\n\n".join(snippets)

        base += "\n\nReturn results either as JSON: {\"patches\": [{\"path\":..., \"patched\": ...}], \"notes\":...} or as plain text with markers '--- FILE: <path> ---' followed by patched content."

        return base

    def _build_prompt_with_snippets(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        feedback: Optional[ValidationReport] = None,
    ) -> str:
        """Build prompt with selected file snippets, template-file support, and optional validator feedback.

        Args:
            cycle: CycleSpec model with id, graph, files.
            description: CycleDescription model with 'text'.
            feedback: Optional ValidationReport for retry.
        """
        file_paths = cycle.get_file_paths()

        # Prepare file content snippets using shared utility for relevant regions
        snippets = []
        cycle_dict = cycle.model_dump()
        for f in cycle.files:
            content = f.content or ""
            snippet = select_relevant_snippet(content, f.path, cycle_dict, self.max_file_chars)
            snippets.append(f"--- FILE: {f.path} ---\n{snippet}")

        file_snippets = "\n\n".join(snippets) if snippets else ""
        
        # Get RAG context for refactoring guidance
        rag_context = self._get_rag_context(cycle)

        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            result = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                file_snippets=file_snippets,
                rag_context=rag_context,
            )
            # Append snippets if template didn't include them
            contains_file_blocks = any((f"--- FILE: {p}" in result) for p in file_paths)
            if file_snippets and "{file_snippets}" not in tpl and not contains_file_blocks:
                result = result + "\n\n" + file_snippets
            
            # Append RAG context if template didn't include it
            if rag_context and "{rag_context}" not in tpl:
                result = result + "\n\n--- REFERENCE MATERIALS ---\n" + rag_context
            
            # Append feedback if provided
            if feedback:
                result += "\n\n## Previous Attempt Feedback\n"
                result += "Your previous proposal was rejected. Please address these issues:\n"
                for issue in feedback.issues:
                    line_info = f" (line {issue.line})" if issue.line else ""
                    result += f"- {issue.path}{line_info}: {issue.comment}\n"
                for suggestion in feedback.suggestions:
                    result += f"- Suggestion: {suggestion}\n"
            return result

        base = f"You are a refactoring assistant. The cyclic dependency: {description.text}.\nFiles: {', '.join(file_paths)}.\nPlease propose refactorings and return patched file contents."

        if file_snippets:
            base += "\n\n" + file_snippets
        
        # Include RAG context for reference
        if rag_context:
            base += "\n\n--- REFERENCE MATERIALS ---\n" + rag_context

        # Include validator feedback if this is a retry
        if feedback:
            base += "\n\n## Previous Attempt Feedback\n"
            base += "Your previous proposal was rejected. Please address these issues:\n"
            for issue in feedback.issues:
                line_info = f" (line {issue.line})" if issue.line else ""
                base += f"- {issue.path}{line_info}: {issue.comment}\n"
            for suggestion in feedback.suggestions:
                base += f"- Suggestion: {suggestion}\n"

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

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        validator_feedback: Optional[Union[ValidationReport, Dict[str, Any]]] = None,
        prompt: str = None,
    ) -> AgentResult:
        """Propose patches to break the cycle.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            validator_feedback: Optional ValidationReport for retry loop.
            prompt: Optional additional instructions.

        Returns:
            AgentResult with RefactorProposal output.
        """
        logger.info(f"RefactorAgent.run() starting, has_feedback={validator_feedback is not None}")

        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            logger.error("Missing description input")
            return AgentResult(status="error", output=None, logs="Missing description")
        if isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if validator_feedback is not None and isinstance(validator_feedback, dict):
            validator_feedback = ValidationReport.model_validate(validator_feedback)
            logger.info(f"Retry with feedback: {len(validator_feedback.issues)} issues, {len(validator_feedback.suggestions)} suggestions")

        # Use enhanced prompt builder that includes selected file snippets and feedback
        prompt_text = self._build_prompt_with_snippets(cycle_spec, description, validator_feedback)
        logger.debug(f"Built refactor prompt with {len(prompt_text)} chars")

        # If no LLM available just return original files as no-op patches
        if self.llm is None:
            logger.info("No LLM provided, returning no-op patches (original files unchanged)")
            patches = [
                Patch(path=f.path, original=f.content, patched=f.content, diff="")
                for f in cycle_spec.files
            ]
            proposal = RefactorProposal(patches=patches, rationale="No-op (no LLM provided)", llm_response=None)
            return AgentResult(status="success", output=proposal.model_dump())

        try:
            logger.info("Calling LLM for refactor proposal")
            llm_response = call_llm(self.llm, prompt_text)
            logger.debug(f"LLM response received, length: {len(str(llm_response))} chars")

            # Convert files to dict for _infer_patches compatibility
            files_dict = [{"path": f.path, "content": f.content} for f in cycle_spec.files]
            inferred = self._infer_patches(llm_response, files_dict)
            logger.debug(f"Inferred {len(inferred)} patches from LLM response")

            # Build final patches list merging originals with patched content (if present)
            patches_out = []
            files_changed = 0
            for f in cycle_spec.files:
                path = f.path
                original = f.content or ""
                patched_entry = next((p for p in inferred if p.get("path") == path), None)
                if patched_entry is None:
                    # try basename match
                    basename = path.split("/")[-1]
                    patched_entry = next((p for p in inferred if p.get("path") == basename), None)

                patched = patched_entry.get("patched") if patched_entry else original

                diff = self._make_unified_diff(original, patched, path) if patched != original else ""
                if diff:
                    files_changed += 1
                    logger.debug(f"File changed: {path} ({len(diff)} chars diff)")

                patches_out.append(Patch(path=path, original=original, patched=patched, diff=diff))

            logger.info(f"RefactorAgent completed: {files_changed}/{len(patches_out)} files changed")
            proposal = RefactorProposal(
                patches=patches_out,
                rationale="LLM produced proposal",
                llm_response=llm_response if isinstance(llm_response, str) else json.dumps(llm_response),
            )
            return AgentResult(status="success", output=proposal.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(status="error", output=None, logs=f"LLM call failed: {e}")
