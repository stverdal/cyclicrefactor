from typing import Dict, Any, List, Optional, Union
import json
import re
from .agent_base import Agent, AgentResult
from .llm_utils import call_llm
from utils.prompt_loader import load_template, safe_format
from utils.logging import get_logger
from models.schemas import (
    CycleSpec,
    CycleDescription,
    RefactorProposal,
    ValidationReport,
    ValidationIssue,
)

logger = get_logger("validator")


class ValidatorAgent(Agent):
    """Validates refactor proposals against the cycle/description and returns approval or feedback.

    When an LLM is provided the validator asks the model to review the proposal
    and decide whether the cycle is addressed. If not, it returns structured
    feedback with inline comments so the refactor agent can retry.
    """

    name = "validator"
    version = "0.3"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        linters: Dict[str, str] = None,
        test_command: str = None,
        max_file_chars: int = 4000,
    ):
        """
        Args:
            llm: Optional LLM client for semantic review.
            prompt_template: Path to prompt file or inline template string.
            linters: Mapping of linter name to shell command (future integration).
            test_command: Shell command to run tests (future integration).
            max_file_chars: Truncation limit per file snippet.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.linters = linters or {}
        self.test_command = test_command
        self.max_file_chars = max_file_chars

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        proposal: RefactorProposal,
    ) -> str:
        """Construct a prompt for the LLM to review the refactor proposal."""
        file_paths = cycle.get_file_paths()

        # Build patched file snippets
        patch_snippets = []
        for p in proposal.patches:
            content = p.patched or ""
            if len(content) > self.max_file_chars:
                content = content[: self.max_file_chars] + "\n...[truncated]"
            patch_snippets.append(f"--- PATCHED FILE: {p.path} ---\n{content}")

        patched_files_text = "\n\n".join(patch_snippets) if patch_snippets else "(no patches)"

        # If we have a template file/path, load it
        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            return safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                patched_files=patched_files_text,
            )

        # Default prompt when no template provided
        base = f"""You are a code-review assistant validating a refactor proposal intended to break a cyclic dependency.

## Cycle Information
- ID: {cycle.id}
- Graph: {json.dumps(cycle.graph.model_dump())}
- Affected files: {', '.join(file_paths)}

## Description from describer agent
{description.text}

## Proposed Patches
{patched_files_text}

## Your Task
1. Determine whether the patches effectively break or reduce the cycle described above.
2. Check for obvious errors, missing imports, or regressions.
3. Decide: APPROVED if the refactor is acceptable, or NEEDS_REVISION otherwise.

## Output Format (strict JSON)
{{
  "decision": "APPROVED" | "NEEDS_REVISION",
  "summary": "<one-sentence verdict>",
  "issues": [
    {{"path": "...", "line": <n or null>, "comment": "<what's wrong or could be improved>"}}
  ],
  "suggestions": ["<concrete next-step if revision needed>"]
}}

Only output the JSON object, nothing else.
"""
        return base

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse the LLM JSON response; return dict or fallback."""
        # Try to extract JSON from the response (model may wrap in markdown)
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: treat entire text as notes
        return {
            "decision": "NEEDS_REVISION",
            "summary": "Could not parse LLM response; manual review required.",
            "issues": [],
            "suggestions": [],
            "raw_response": text,
        }

    # -------------------------------------------------------------------------
    # Rule-based checks (used when no LLM or as additional layer)
    # -------------------------------------------------------------------------

    def _rule_based_checks(
        self, cycle: CycleSpec, proposal: RefactorProposal
    ) -> List[ValidationIssue]:
        """Perform lightweight deterministic checks on the proposal."""
        issues: List[ValidationIssue] = []
        known_paths = {f.path for f in cycle.files}

        for p in proposal.patches:
            path = p.path
            # Check patch targets known file
            if path not in known_paths and path.split("/")[-1] not in {
                pth.split("/")[-1] for pth in known_paths
            }:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="Patch targets unknown file")
                )
            # Check non-empty patched content
            if not p.patched:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="Patched content is empty")
                )
            # Check diff exists (i.e., something changed)
            if p.original == p.patched:
                issues.append(
                    ValidationIssue(path=path, line=None, comment="No changes detected in patch")
                )

        return issues

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any]] = None,
        proposal: Union[RefactorProposal, Dict[str, Any]] = None,
    ) -> AgentResult:
        """Validate the proposal and return approval or feedback.

        Args:
            cycle_spec: CycleSpec model or dict with id, graph, files.
            description: CycleDescription model or dict from describer.
            proposal: RefactorProposal model or dict from refactor agent.

        Returns:
            AgentResult with ValidationReport output.
        """
        logger.info("ValidatorAgent.run() starting")

        # Convert inputs to models if needed
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description is None:
            description = CycleDescription(text="")
        elif isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if proposal is None:
            logger.error("Missing proposal input")
            return AgentResult(
                status="error", output=None, logs="Missing proposal"
            )
        if isinstance(proposal, dict):
            proposal = RefactorProposal.model_validate(proposal)

        logger.debug(f"Validating proposal with {len(proposal.patches)} patches")

        # Rule-based checks first
        rule_issues = self._rule_based_checks(cycle_spec, proposal)
        logger.debug(f"Rule-based checks found {len(rule_issues)} issues")

        # If no LLM, return rule-based result
        if self.llm is None:
            approved = len(rule_issues) == 0
            logger.info(f"No LLM provided, rule-based validation: approved={approved}")
            report = ValidationReport(
                approved=approved,
                decision="APPROVED" if approved else "NEEDS_REVISION",
                summary="Rule-based validation only (no LLM provided).",
                issues=rule_issues,
                suggestions=["Provide an LLM for semantic review."] if rule_issues else [],
            )
            return AgentResult(status="success", output=report.model_dump())

        # LLM-based review
        prompt = self._build_prompt(cycle_spec, description, proposal)
        logger.debug(f"Built validation prompt with {len(prompt)} chars")

        try:
            logger.info("Calling LLM for semantic validation")
            response = call_llm(self.llm, prompt)
            text = response if isinstance(response, str) else json.dumps(response)
            parsed = self._parse_llm_response(text)
            logger.debug(f"LLM decision: {parsed.get('decision', 'unknown')}")

            # Merge rule-based issues with parsed issues
            parsed_issues = [
                ValidationIssue.model_validate(i) if isinstance(i, dict) else i
                for i in parsed.get("issues", [])
            ]
            all_issues = rule_issues + parsed_issues
            decision = parsed.get("decision", "NEEDS_REVISION")
            approved = decision == "APPROVED" and len(all_issues) == 0

            logger.info(f"ValidatorAgent completed: decision={decision}, approved={approved}, issues={len(all_issues)}")
            report = ValidationReport(
                approved=approved,
                decision=decision if approved else "NEEDS_REVISION",
                summary=parsed.get("summary", ""),
                issues=all_issues,
                suggestions=parsed.get("suggestions", []),
            )
            return AgentResult(status="success", output=report.model_dump())

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return AgentResult(
                status="error", output=None, logs=f"LLM call failed: {e}"
            )
