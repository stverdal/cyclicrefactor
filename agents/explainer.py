from typing import Dict, Any, Optional, Union
import json
import re
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.prompt_loader import load_template, safe_format
from models.schemas import (
    CycleSpec,
    CycleDescription,
    RefactorProposal,
    ValidationReport,
    Explanation,
)


class ExplainerAgent(Agent):
    """Generates a human-readable explanation of the refactoring after validation passes.

    This agent runs at the end of the pipeline to summarize:
    - What cycle was addressed
    - What changes were made
    - Why the changes break/reduce the cycle
    - Any caveats or follow-up recommendations
    """

    name = "explainer"
    version = "0.2"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        max_diff_chars: int = 8000,
    ):
        """
        Args:
            llm: Optional LLM client for generating explanations.
            prompt_template: Path to prompt file or inline template string.
            max_diff_chars: Max characters of diff content to include in prompt.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_diff_chars = max_diff_chars

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        proposal: RefactorProposal,
        validation: ValidationReport,
    ) -> str:
        """Construct prompt for the LLM to generate a refactor explanation."""
        file_paths = cycle.get_file_paths()

        # Build diff summary
        diff_parts = []
        total_len = 0
        for p in proposal.patches:
            diff = p.diff or ""
            if diff:
                diff_parts.append(f"### {p.path}\n```diff\n{diff}\n```")
                total_len += len(diff)
                if total_len > self.max_diff_chars:
                    diff_parts.append("...[additional diffs truncated]")
                    break

        diffs_text = "\n\n".join(diff_parts) if diff_parts else "(no diffs available)"

        # Load template if provided
        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            return safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                diffs=diffs_text,
                validation_summary=validation.summary,
            )

        # Default prompt
        return f"""You are a technical writer summarizing a code refactor for a development team.

## Cycle Information
- ID: {cycle.id}
- Dependency graph: {json.dumps(cycle.graph.model_dump())}
- Affected files: {', '.join(file_paths)}

## Original Problem (from describer)
{description.text}

## Changes Made (diffs)
{diffs_text}

## Validation Result
{validation.summary or 'Approved'}

## Your Task
Write a clear, concise explanation (2-4 paragraphs) covering:
1. **Problem**: What cyclic dependency existed and why it was problematic.
2. **Solution**: What changes were made to break/reduce the cycle.
3. **Impact**: How the codebase is improved (maintainability, testability, etc.).
4. **Caveats/Follow-up**: Any remaining risks or recommended next steps.

Keep the tone professional and suitable for a pull request description or technical changelog.

## Output Format
Return a JSON object:
{{
  "title": "<short title for the refactor, e.g., 'Break A↔B cycle via interface extraction'>",
  "summary": "<1-2 sentence TL;DR>",
  "explanation": "<full explanation in markdown>",
  "impact": ["<list of impacts/benefits>"],
  "followup": ["<optional next steps or caveats>"]
}}

Only output the JSON object.
"""

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response; return dict or fallback."""
        import re
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: treat text as explanation
        return {
            "title": "Refactor Explanation",
            "summary": "See explanation below.",
            "explanation": text,
            "impact": [],
            "followup": [],
        }

    # -------------------------------------------------------------------------
    # Fallback (no LLM)
    # -------------------------------------------------------------------------

    def _generate_fallback(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        proposal: RefactorProposal,
    ) -> Explanation:
        """Generate a basic explanation without an LLM."""
        file_paths = cycle.get_file_paths()
        nodes = cycle.graph.nodes
        changed = [p.path for p in proposal.patches if p.diff]

        explanation_text = f"""## Refactor Summary

A cyclic dependency involving {len(nodes)} component(s) ({', '.join(nodes)}) was addressed.

### Files Modified
{chr(10).join('- ' + p for p in changed) if changed else '- No files were modified.'}

### Original Problem
{description.text or 'N/A'}

### Next Steps
- Review the generated patches before merging.
- Run tests to ensure no regressions.
"""
        return Explanation(
            title=f"Break cycle: {' ↔ '.join(nodes[:3])}{'...' if len(nodes) > 3 else ''}",
            summary=f"Refactored {len(changed)} file(s) to address cyclic dependency.",
            explanation=explanation_text,
            impact=["Reduced coupling between modules."],
            followup=["Run full test suite.", "Consider adding integration tests."],
        )

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        proposal: Union[RefactorProposal, Dict[str, Any]] = None,
        validation: Union[ValidationReport, Dict[str, Any]] = None,
    ) -> AgentResult:
        """Generate an explanation of the approved refactor.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            proposal: RefactorProposal model or dict from refactor agent.
            validation: ValidationReport model or dict from validator.

        Returns:
            AgentResult with Explanation output.
        """
        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            description = CycleDescription(text="")
        elif isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if proposal is None:
            return AgentResult(
                status="error", output=None, logs="Missing proposal"
            )
        if isinstance(proposal, dict):
            proposal = RefactorProposal.model_validate(proposal)
        if validation is None:
            validation = ValidationReport(approved=True, summary="")
        elif isinstance(validation, dict):
            validation = ValidationReport.model_validate(validation)

        # If validation didn't pass, we shouldn't be here, but handle gracefully
        if not validation.approved:
            return AgentResult(
                status="error",
                output=None,
                logs="Cannot generate explanation for unapproved proposal",
            )

        # If no LLM, use fallback
        if self.llm is None:
            result = self._generate_fallback(cycle_spec, description, proposal)
            return AgentResult(status="success", output=result.model_dump())

        # LLM-based explanation
        prompt = self._build_prompt(cycle_spec, description, proposal, validation)

        try:
            response = call_llm(self.llm, prompt)
            text = response if isinstance(response, str) else json.dumps(response)
            parsed = self._parse_llm_response(text)
            # Validate through Explanation model
            result = Explanation.model_validate(parsed)
            return AgentResult(status="success", output=result.model_dump())
        except Exception as e:
            return AgentResult(
                status="error", output=None, logs=f"LLM call failed: {e}"
            )
