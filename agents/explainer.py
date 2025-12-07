from typing import Dict, Any, Optional, Union, List, Tuple
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
    Explanation,
)

logger = get_logger("explainer")

# Optional RAG import
try:
    from rag.rag_service import RAGService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.debug("RAG service not available for explainer")


class ExplainerAgent(Agent):
    """Generates a human-readable explanation of the refactoring after validation passes.

    This agent runs at the end of the pipeline to summarize:
    - What cycle was addressed
    - What changes were made
    - Why the changes break/reduce the cycle
    - Any caveats or follow-up recommendations
    
    Optionally integrates with RAG to cite best practices from indexed documents.
    """

    name = "explainer"
    version = "0.3"

    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        max_diff_chars: int = 6000,
        rag_service: Optional[Any] = None,
    ):
        """
        Args:
            llm: Optional LLM client for generating explanations.
            prompt_template: Path to prompt file or inline template string.
            max_diff_chars: Max characters of diff content to include in prompt.
            rag_service: Optional RAGService for retrieving best practice context.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_diff_chars = max_diff_chars
        self.rag_service = rag_service

    # -------------------------------------------------------------------------
    # Diff analysis helpers
    # -------------------------------------------------------------------------

    def _analyze_diffs(self, proposal: RefactorProposal) -> Dict[str, Any]:
        """Parse diffs to extract structured change information.
        
        Returns:
            Dict with keys: files_modified, interfaces_added, classes_moved,
            methods_extracted, imports_changed, etc.
        """
        analysis = {
            "files_modified": [],
            "interfaces_added": [],
            "classes_added": [],
            "methods_extracted": [],
            "imports_added": [],
            "imports_removed": [],
            "lines_added": 0,
            "lines_removed": 0,
        }
        
        for patch in proposal.patches:
            if not patch.diff:
                continue
            
            analysis["files_modified"].append(patch.path)
            
            # Parse diff to count lines
            for line in patch.diff.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    analysis["lines_added"] += 1
                    
                    # Detect new interfaces
                    if re.search(r'^\+\s*(public\s+)?interface\s+\w+', line):
                        match = re.search(r'interface\s+(\w+)', line)
                        if match:
                            analysis["interfaces_added"].append(match.group(1))
                            logger.debug(f"Detected new interface: {match.group(1)}")
                    
                    # Detect new classes
                    if re.search(r'^\+\s*(public\s+)?(abstract\s+)?class\s+\w+', line):
                        match = re.search(r'class\s+(\w+)', line)
                        if match:
                            analysis["classes_added"].append(match.group(1))
                            logger.debug(f"Detected new class: {match.group(1)}")
                    
                    # Detect new imports (C#)
                    if re.search(r'^\+\s*using\s+[\w.]+;', line):
                        match = re.search(r'using\s+([\w.]+);', line)
                        if match:
                            analysis["imports_added"].append(match.group(1))
                    
                    # Detect new imports (Python)
                    if re.search(r'^\+\s*(from\s+\S+\s+)?import\s+', line):
                        analysis["imports_added"].append(line.strip()[1:].strip())
                    
                elif line.startswith('-') and not line.startswith('---'):
                    analysis["lines_removed"] += 1
                    
                    # Detect removed imports
                    if re.search(r'^-\s*using\s+[\w.]+;', line):
                        match = re.search(r'using\s+([\w.]+);', line)
                        if match:
                            analysis["imports_removed"].append(match.group(1))
        
        logger.info(f"Diff analysis: {len(analysis['files_modified'])} files, "
                   f"+{analysis['lines_added']}/-{analysis['lines_removed']} lines, "
                   f"{len(analysis['interfaces_added'])} new interfaces")
        
        return analysis

    def _get_rag_context(self, cycle: CycleSpec, diff_analysis: Dict[str, Any]) -> str:
        """Query RAG for relevant best practices based on the refactoring pattern."""
        if not self.rag_service:
            return ""
        
        # Build query based on detected patterns
        patterns = []
        if diff_analysis.get("interfaces_added"):
            patterns.append("interface extraction dependency inversion")
        if diff_analysis.get("classes_added"):
            patterns.append("class extraction single responsibility")
        if diff_analysis.get("imports_removed"):
            patterns.append("decoupling reducing dependencies")
        
        if not patterns:
            patterns.append("breaking cyclic dependencies refactoring")
        
        query = f"Best practices for: {', '.join(patterns)}"
        logger.info(f"RAG query for explainer: {query}")
        
        try:
            docs = self.rag_service.query_with_scores(query, k=2)
            if docs:
                context_parts = []
                for doc, score in docs:
                    source = doc.metadata.get("source", "unknown")
                    context_parts.append(f"[{source}]: {doc.page_content[:500]}")
                    logger.debug(f"RAG retrieved from {source} (score: {score:.3f})")
                return "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
        
        return ""

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        cycle: CycleSpec,
        description: CycleDescription,
        proposal: RefactorProposal,
        validation: ValidationReport,
        diff_analysis: Dict[str, Any],
        rag_context: str = "",
    ) -> str:
        """Construct prompt for the LLM to generate a refactor explanation."""
        file_paths = cycle.get_file_paths()

        # Build diff summary (compact format for limited context)
        diff_parts = []
        total_len = 0
        for p in proposal.patches:
            diff = p.diff or ""
            if diff:
                # Truncate individual diffs if too long
                if len(diff) > 1500:
                    diff = diff[:1500] + "\n... [diff truncated]"
                diff_parts.append(f"### {p.path}\n```diff\n{diff}\n```")
                total_len += len(diff)
                if total_len > self.max_diff_chars:
                    remaining = len(proposal.patches) - len(diff_parts)
                    if remaining > 0:
                        diff_parts.append(f"... [{remaining} more file(s) not shown]")
                    break

        diffs_text = "\n\n".join(diff_parts) if diff_parts else "(no diffs available)"
        
        # Build change summary from analysis
        change_summary = []
        if diff_analysis["interfaces_added"]:
            change_summary.append(f"New interfaces: {', '.join(diff_analysis['interfaces_added'])}")
        if diff_analysis["classes_added"]:
            change_summary.append(f"New classes: {', '.join(diff_analysis['classes_added'])}")
        if diff_analysis["imports_removed"]:
            change_summary.append(f"Removed imports: {len(diff_analysis['imports_removed'])}")
        change_summary.append(f"Lines: +{diff_analysis['lines_added']}/-{diff_analysis['lines_removed']}")
        change_text = " | ".join(change_summary)

        # Load template if provided
        if self.prompt_template:
            tpl = load_template(self.prompt_template)
            prompt = safe_format(
                tpl,
                id=cycle.id,
                graph=json.dumps(cycle.graph.model_dump()),
                files=", ".join(file_paths),
                description=description.text,
                diffs=diffs_text,
                validation_summary=validation.summary,
            )
            # Append extra context
            if change_text:
                prompt += f"\n\n## Change Summary\n{change_text}"
            if rag_context:
                prompt += f"\n\n## Reference (from architecture docs)\n{rag_context}"
            return prompt

        # Default compact prompt
        prompt = f"""Summarize this refactor for a pull request.

## Cycle: {cycle.id}
Nodes: {', '.join(cycle.graph.nodes)}
Files: {', '.join(file_paths)}

## Problem
{description.text[:1000] if description.text else 'N/A'}

## Changes
{change_text}

{diffs_text}
"""
        
        if rag_context:
            prompt += f"""
## Best Practice Reference
{rag_context[:800]}
"""
        
        prompt += """
## Task
Write a JSON explanation:
{
  "title": "<short title>",
  "summary": "<1-2 sentences>",
  "explanation": "<2-3 paragraphs in markdown>",
  "impact": ["<benefits>"],
  "followup": ["<next steps>"]
}
Only output the JSON.
"""
        return prompt

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response; return dict or fallback."""
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
        diff_analysis: Dict[str, Any],
    ) -> Explanation:
        """Generate a structured explanation without an LLM using diff analysis."""
        nodes = cycle.graph.nodes
        changed = diff_analysis.get("files_modified", [])
        
        # Determine the refactoring pattern used
        pattern = "code modification"
        if diff_analysis.get("interfaces_added"):
            pattern = f"interface extraction ({', '.join(diff_analysis['interfaces_added'][:2])})"
        elif diff_analysis.get("classes_added"):
            pattern = f"class extraction ({', '.join(diff_analysis['classes_added'][:2])})"
        elif diff_analysis.get("imports_removed"):
            pattern = "dependency removal"
        
        # Build structured explanation
        explanation_parts = [
            "## Summary",
            f"This refactor addresses a cyclic dependency involving {len(nodes)} component(s): "
            f"{', '.join(nodes[:4])}{'...' if len(nodes) > 4 else ''}.",
            "",
            "## Changes Made",
        ]
        
        if changed:
            explanation_parts.append(f"Modified {len(changed)} file(s):")
            for path in changed[:5]:
                explanation_parts.append(f"- `{path}`")
            if len(changed) > 5:
                explanation_parts.append(f"- ... and {len(changed) - 5} more")
        
        if diff_analysis.get("interfaces_added"):
            explanation_parts.append("")
            explanation_parts.append("**New interfaces introduced:**")
            for iface in diff_analysis["interfaces_added"]:
                explanation_parts.append(f"- `{iface}` - enables dependency inversion")
        
        if diff_analysis.get("classes_added"):
            explanation_parts.append("")
            explanation_parts.append("**New classes created:**")
            for cls in diff_analysis["classes_added"]:
                explanation_parts.append(f"- `{cls}`")
        
        explanation_parts.extend([
            "",
            "## Statistics",
            f"- Lines added: {diff_analysis.get('lines_added', 0)}",
            f"- Lines removed: {diff_analysis.get('lines_removed', 0)}",
        ])
        
        if description.text:
            explanation_parts.extend([
                "",
                "## Original Problem",
                description.text[:500] + ("..." if len(description.text) > 500 else ""),
            ])
        
        explanation_text = "\n".join(explanation_parts)
        
        # Determine impacts
        impacts = ["Reduced coupling between modules"]
        if diff_analysis.get("interfaces_added"):
            impacts.append("Enables dependency injection and easier testing")
            impacts.append("Follows Interface Segregation Principle")
        if diff_analysis.get("imports_removed"):
            impacts.append(f"Removed {len(diff_analysis['imports_removed'])} unnecessary import(s)")
        
        # Build title
        title = f"Break cycle via {pattern}"
        if len(title) > 60:
            title = f"Break {' ↔ '.join(nodes[:2])} cycle"
        
        logger.info(f"Generated fallback explanation: {title}")
        
        return Explanation(
            title=title,
            summary=f"Refactored {len(changed)} file(s) to break cyclic dependency using {pattern}.",
            explanation=explanation_text,
            impact=impacts,
            followup=[
                "Review the generated patches before merging.",
                "Run full test suite to ensure no regressions.",
                "Consider adding integration tests for the new interfaces.",
            ],
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
        logger.info("ExplainerAgent.run() starting")

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
        if validation is None:
            validation = ValidationReport(approved=True, summary="")
        elif isinstance(validation, dict):
            validation = ValidationReport.model_validate(validation)

        # If validation didn't pass, we shouldn't be here, but handle gracefully
        if not validation.approved:
            logger.warning("Explainer called with unapproved proposal, skipping")
            return AgentResult(
                status="error",
                output=None,
                logs="Cannot generate explanation for unapproved proposal",
            )

        # Analyze diffs for structured information
        diff_analysis = self._analyze_diffs(proposal)
        logger.info(f"Diff analysis complete: {len(diff_analysis['files_modified'])} files, "
                   f"{len(diff_analysis['interfaces_added'])} interfaces, "
                   f"{len(diff_analysis['classes_added'])} classes")

        # If no LLM, use enhanced fallback
        if self.llm is None:
            logger.info("No LLM provided, using enhanced fallback explanation generator")
            result = self._generate_fallback(cycle_spec, description, proposal, diff_analysis)
            logger.debug(f"Generated fallback explanation: {result.title}")
            return AgentResult(status="success", output=result.model_dump())

        # Get RAG context if available
        rag_context = self._get_rag_context(cycle_spec, diff_analysis)
        if rag_context:
            logger.info(f"RAG context retrieved ({len(rag_context)} chars)")

        # LLM-based explanation
        prompt = self._build_prompt(
            cycle_spec, description, proposal, validation, diff_analysis, rag_context
        )
        logger.debug(f"Built explanation prompt with {len(prompt)} chars")

        try:
            logger.info("Calling LLM for explanation generation")
            response = call_llm(self.llm, prompt)
            text = response if isinstance(response, str) else json.dumps(response)
            parsed = self._parse_llm_response(text)
            # Validate through Explanation model
            result = Explanation.model_validate(parsed)
            logger.info(f"ExplainerAgent completed: {result.title}")
            return AgentResult(status="success", output=result.model_dump())
        except Exception as e:
            logger.error(f"LLM call failed: {e}, falling back to template-based explanation")
            # Fallback to template-based on LLM failure
            result = self._generate_fallback(cycle_spec, description, proposal, diff_analysis)
            return AgentResult(
                status="success", 
                output=result.model_dump(),
                logs=f"LLM call failed, used fallback: {e}"
            )
