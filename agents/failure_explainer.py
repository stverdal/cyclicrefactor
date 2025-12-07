"""Failure Explainer Agent - Generates human-readable guidance when patches fail.

This agent is invoked when the pipeline cannot successfully apply patches,
providing:
1. What was attempted and why it failed
2. Analysis of the failure patterns
3. Actionable recommendations for human operators
4. The raw LLM output for manual application

This is critical for smaller open-source LLMs where patch application may fail
more frequently due to:
- Hallucinated code that doesn't match actual file content
- Incorrect whitespace/indentation
- Partial context leading to wrong assumptions
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
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
    Patch,
    RevertedFile,
)

logger = get_logger("failure_explainer")


@dataclass
class FailurePattern:
    """Detected pattern in patch failures."""
    pattern_type: str  # "search_not_found", "syntax_error", "validation_failed", etc.
    count: int
    affected_files: List[str]
    description: str
    likely_cause: str
    recommendation: str


class FailureExplainerAgent(Agent):
    """Generates explanations and guidance when patches fail to apply.
    
    This agent analyzes:
    1. What the LLM tried to do (from llm_response)
    2. Why it failed (from warnings, validation_issues, revert_reasons)
    3. Patterns across multiple failures
    4. Historical context (previous attempts, failed strategies)
    
    Output includes:
    - Executive summary of the failure
    - Detailed analysis per file
    - Extracted LLM suggestions (human-readable format)
    - Manual remediation steps
    """
    
    name = "failure_explainer"
    version = "0.1"
    
    # Known failure patterns and their remediation advice
    FAILURE_PATTERNS = {
        "search_not_found": {
            "indicators": ["search text not found", "no match found", "not found in file"],
            "likely_cause": "The LLM generated code that doesn't match the actual file content. This is common with smaller models that hallucinate code or misremember exact syntax.",
            "recommendation": "Compare the LLM's search text with the actual file. Look for differences in whitespace, variable names, or code structure. The LLM's intent may be correct even if the exact text is wrong.",
        },
        "low_confidence": {
            "indicators": ["low confidence", "marginal confidence", "fuzzy match"],
            "likely_cause": "The matching algorithm found a similar but not exact match. The LLM's output was close but had minor differences.",
            "recommendation": "Review the fuzzy match - it may be acceptable. Consider enabling 'allow_low_confidence' in config if the matches look reasonable.",
        },
        "syntax_error": {
            "indicators": ["syntax error", "parse error", "unexpected token", "missing semicolon", "unclosed"],
            "likely_cause": "The LLM generated syntactically invalid code, possibly due to truncation or incomplete understanding of the language syntax.",
            "recommendation": "Check for truncated output, missing closing braces/brackets, or incomplete statements. The LLM's approach may be sound but the output incomplete.",
        },
        "truncation": {
            "indicators": ["truncated", "incomplete", "cut off", "..."],
            "likely_cause": "The LLM's output was cut off due to context window limits or max token settings.",
            "recommendation": "Try increasing max_tokens, or ask the LLM to focus on smaller, more targeted changes instead of full file rewrites.",
        },
        "cycle_not_broken": {
            "indicators": ["cycle not broken", "cycle still exists", "dependency remains"],
            "likely_cause": "The proposed changes don't actually break the cyclic dependency - they may move code around without changing the import structure.",
            "recommendation": "Focus on which specific import creates the cycle. The goal is to remove or invert that dependency, not just refactor the code.",
        },
        "compile_error": {
            "indicators": ["compile error", "compilation failed", "build error", "type error"],
            "likely_cause": "The patched code doesn't compile, likely due to missing imports, type mismatches, or references to undefined symbols.",
            "recommendation": "Check the compiler errors carefully - they usually point to the exact issue. Common fixes: add missing imports, fix type annotations, ensure all referenced classes exist.",
        },
    }
    
    def __init__(
        self,
        llm=None,
        prompt_template: str = None,
        include_llm_suggestions: bool = True,
    ):
        """
        Args:
            llm: Optional LLM for generating enhanced explanations.
            prompt_template: Path to prompt template.
            include_llm_suggestions: Whether to ask LLM for additional suggestions.
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.include_llm_suggestions = include_llm_suggestions
    
    # -------------------------------------------------------------------------
    # Failure Analysis
    # -------------------------------------------------------------------------
    
    def _detect_failure_patterns(
        self,
        proposal: RefactorProposal,
        validation: Optional[ValidationReport],
    ) -> List[FailurePattern]:
        """Analyze failures to detect common patterns."""
        patterns_found: Dict[str, FailurePattern] = {}
        
        # Collect all warning/error messages
        all_messages = []
        file_to_messages: Dict[str, List[str]] = {}
        
        for patch in proposal.patches:
            path = patch.path
            file_messages = []
            
            # Collect from various sources
            file_messages.extend(patch.warnings or [])
            file_messages.extend(patch.validation_issues or [])
            if patch.revert_reason:
                file_messages.append(patch.revert_reason)
            
            file_to_messages[path] = file_messages
            all_messages.extend(file_messages)
        
        for rf in proposal.reverted_files:
            path = rf.path
            file_messages = file_to_messages.get(path, [])
            file_messages.extend(rf.warnings or [])
            if rf.reason:
                file_messages.append(rf.reason)
            file_to_messages[path] = file_messages
            all_messages.extend(rf.warnings or [])
        
        # Add validation issues
        if validation and validation.issues:
            for issue in validation.issues:
                msg = f"{issue.issue_type}: {issue.comment}"
                all_messages.append(msg)
                path = issue.path
                if path:
                    file_to_messages.setdefault(path, []).append(msg)
        
        # Match against known patterns
        all_text = " ".join(all_messages).lower()
        
        for pattern_name, pattern_info in self.FAILURE_PATTERNS.items():
            for indicator in pattern_info["indicators"]:
                if indicator.lower() in all_text:
                    # Find affected files
                    affected = []
                    for path, messages in file_to_messages.items():
                        msg_text = " ".join(messages).lower()
                        if indicator.lower() in msg_text:
                            affected.append(path)
                    
                    if pattern_name not in patterns_found:
                        patterns_found[pattern_name] = FailurePattern(
                            pattern_type=pattern_name,
                            count=1,
                            affected_files=affected,
                            description=indicator,
                            likely_cause=pattern_info["likely_cause"],
                            recommendation=pattern_info["recommendation"],
                        )
                    else:
                        patterns_found[pattern_name].count += 1
                        for f in affected:
                            if f not in patterns_found[pattern_name].affected_files:
                                patterns_found[pattern_name].affected_files.append(f)
                    break  # Only match once per pattern type
        
        return list(patterns_found.values())
    
    def _extract_llm_intent(self, proposal: RefactorProposal) -> Dict[str, Any]:
        """Extract the LLM's intended changes from the raw response."""
        intent = {
            "strategy": None,
            "reasoning": None,
            "per_file_changes": {},
            "raw_patches": [],
        }
        
        if not proposal.llm_response:
            return intent
        
        try:
            # Try to parse as JSON
            text = proposal.llm_response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                intent["strategy"] = parsed.get("strategy_used") or parsed.get("strategy")
                intent["reasoning"] = parsed.get("reasoning")
                
                # Extract per-file changes
                changes = parsed.get("changes") or parsed.get("patches") or []
                for change in changes:
                    path = change.get("path", "unknown")
                    intent["per_file_changes"][path] = {
                        "search_replace": change.get("search_replace", []),
                        "patched": change.get("patched"),
                    }
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Also store raw patches for display
        for patch in proposal.patches:
            if patch.status in ("failed", "reverted", "partial"):
                intent["raw_patches"].append({
                    "path": patch.path,
                    "status": patch.status,
                    "applied": f"{patch.applied_blocks}/{patch.total_blocks}",
                    "warnings": patch.warnings,
                })
        
        return intent
    
    def _generate_per_file_analysis(
        self,
        proposal: RefactorProposal,
        patterns: List[FailurePattern],
    ) -> List[Dict[str, Any]]:
        """Generate detailed analysis for each failed file."""
        file_analyses = []
        
        for patch in proposal.patches:
            if patch.status not in ("failed", "reverted", "partial"):
                continue
            
            analysis = {
                "path": patch.path,
                "status": patch.status,
                "what_was_attempted": "",
                "why_it_failed": "",
                "llm_search_replace": [],
                "manual_steps": [],
            }
            
            # Determine what was attempted
            if patch.total_blocks > 0:
                analysis["what_was_attempted"] = (
                    f"LLM proposed {patch.total_blocks} SEARCH/REPLACE operation(s), "
                    f"but only {patch.applied_blocks} could be matched in the file."
                )
            else:
                analysis["what_was_attempted"] = (
                    "LLM proposed changes but they could not be applied."
                )
            
            # Why it failed
            if patch.warnings:
                analysis["why_it_failed"] = "; ".join(patch.warnings[:3])
            elif patch.revert_reason:
                analysis["why_it_failed"] = patch.revert_reason
            elif patch.validation_issues:
                analysis["why_it_failed"] = "; ".join(patch.validation_issues[:3])
            else:
                analysis["why_it_failed"] = "Unknown failure reason"
            
            # Add manual remediation steps based on patterns
            for pattern in patterns:
                if patch.path in pattern.affected_files:
                    analysis["manual_steps"].append(pattern.recommendation)
            
            file_analyses.append(analysis)
        
        # Also include reverted files not in patches
        reverted_paths = {p.path for p in proposal.patches}
        for rf in proposal.reverted_files:
            if rf.path not in reverted_paths:
                file_analyses.append({
                    "path": rf.path,
                    "status": "reverted",
                    "what_was_attempted": "Changes were attempted but reverted.",
                    "why_it_failed": rf.reason,
                    "llm_search_replace": [],
                    "manual_steps": [],
                })
        
        return file_analyses
    
    def _format_llm_changes_for_human(
        self,
        proposal: RefactorProposal,
    ) -> str:
        """Format the LLM's intended changes in a human-readable way."""
        parts = []
        
        if not proposal.llm_response:
            return "No LLM response available."
        
        parts.append("## What the LLM Tried to Do\n")
        
        # Try to extract and format search/replace operations
        try:
            text = proposal.llm_response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Show reasoning
                if parsed.get("reasoning"):
                    parts.append(f"**Reasoning:** {parsed['reasoning']}\n")
                
                if parsed.get("strategy_used"):
                    parts.append(f"**Strategy:** {parsed['strategy_used']}\n")
                
                # Show changes per file
                changes = parsed.get("changes") or parsed.get("patches") or []
                for change in changes:
                    path = change.get("path", "unknown")
                    parts.append(f"\n### File: `{path}`\n")
                    
                    sr_list = change.get("search_replace", [])
                    if sr_list:
                        for i, sr in enumerate(sr_list):
                            parts.append(f"\n**Change {i + 1}:**")
                            parts.append("\n```")
                            parts.append("FIND:")
                            search = sr.get("search", "")
                            for line in search.split('\n')[:10]:
                                parts.append(f"  {line}")
                            if search.count('\n') > 10:
                                parts.append(f"  ... ({search.count(chr(10)) - 10} more lines)")
                            parts.append("\nREPLACE WITH:")
                            replace = sr.get("replace", "")
                            for line in replace.split('\n')[:10]:
                                parts.append(f"  {line}")
                            if replace.count('\n') > 10:
                                parts.append(f"  ... ({replace.count(chr(10)) - 10} more lines)")
                            parts.append("```\n")
                    
                    elif change.get("patched"):
                        parts.append("(Full file replacement proposed - see raw LLM output)\n")
                
                return "\n".join(parts)
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Could not parse LLM response as JSON: {e}")
        
        # Fallback: show raw response (truncated)
        parts.append("Could not parse structured changes. Raw LLM output:\n")
        parts.append("```")
        raw = proposal.llm_response[:3000]
        parts.append(raw)
        if len(proposal.llm_response) > 3000:
            parts.append(f"\n... ({len(proposal.llm_response) - 3000} more chars)")
        parts.append("```")
        
        return "\n".join(parts)
    
    # -------------------------------------------------------------------------
    # Main Explanation Generation
    # -------------------------------------------------------------------------
    
    def _generate_failure_explanation(
        self,
        cycle: CycleSpec,
        description: Optional[CycleDescription],
        proposal: RefactorProposal,
        validation: Optional[ValidationReport],
        validation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate comprehensive failure explanation."""
        
        # Detect patterns
        patterns = self._detect_failure_patterns(proposal, validation)
        logger.info(f"Detected {len(patterns)} failure patterns")
        
        # Extract LLM intent
        llm_intent = self._extract_llm_intent(proposal)
        
        # Generate per-file analysis
        file_analyses = self._generate_per_file_analysis(proposal, patterns)
        
        # Build summary statistics
        failed_count = len([p for p in proposal.patches if p.status in ("failed", "reverted")])
        total_count = len(proposal.patches)
        partial_count = len([p for p in proposal.patches if p.status == "partial"])
        
        # Executive summary
        if failed_count == total_count:
            summary = f"All {total_count} file(s) failed to patch. The LLM's proposed changes could not be applied."
        elif partial_count > 0:
            summary = f"{failed_count} file(s) failed, {partial_count} partially applied. Some changes may be incomplete."
        else:
            summary = f"{failed_count} of {total_count} file(s) failed to patch."
        
        # Build recommendations
        recommendations = []
        seen_recs = set()
        
        for pattern in patterns:
            if pattern.recommendation not in seen_recs:
                recommendations.append({
                    "issue": pattern.pattern_type.replace("_", " ").title(),
                    "affected_files": pattern.affected_files[:3],
                    "suggestion": pattern.recommendation,
                })
                seen_recs.add(pattern.recommendation)
        
        # Add generic recommendations if few specific ones
        if len(recommendations) < 2:
            recommendations.append({
                "issue": "General",
                "affected_files": [],
                "suggestion": "Review the LLM's intended changes below and apply them manually with adjustments as needed.",
            })
            recommendations.append({
                "issue": "Alternative Approach",
                "affected_files": [],
                "suggestion": "Try running with a larger/different LLM model, or provide more context about the codebase.",
            })
        
        # Format LLM changes for human review
        llm_changes_formatted = self._format_llm_changes_for_human(proposal)
        
        # Build the explanation document
        explanation = {
            "title": "Refactoring Failed - Manual Intervention Required",
            "status": "failed",
            "cycle_id": cycle.id,
            "summary": summary,
            "statistics": {
                "total_files": total_count,
                "failed_files": failed_count,
                "partial_files": partial_count,
                "iterations_attempted": len(validation_history),
            },
            "failure_patterns": [
                {
                    "pattern": p.pattern_type,
                    "description": p.likely_cause,
                    "affected_files": p.affected_files,
                }
                for p in patterns
            ],
            "per_file_analysis": file_analyses,
            "recommendations": recommendations,
            "llm_intent": {
                "strategy": llm_intent.get("strategy"),
                "reasoning": llm_intent.get("reasoning"),
            },
            "llm_changes_for_manual_application": llm_changes_formatted,
            "next_steps": [
                "1. Review the 'What the LLM Tried to Do' section to understand the intended refactoring",
                "2. Compare the LLM's search text with actual file content to identify mismatches",
                "3. Manually apply the changes, adapting for any differences in the actual code",
                "4. Verify that the changes actually break the cycle (check imports)",
                "5. Run tests to ensure no regressions",
            ],
            "raw_llm_response": proposal.llm_response[:5000] if proposal.llm_response else None,
        }
        
        # Add historical context if available
        if validation_history:
            explanation["attempt_history"] = [
                {
                    "iteration": i + 1,
                    "approved": v.get("approved", False),
                    "issues_count": len(v.get("issues", [])),
                }
                for i, v in enumerate(validation_history)
            ]
        
        return explanation
    
    def _generate_markdown_report(self, explanation: Dict[str, Any]) -> str:
        """Generate a human-readable Markdown report from the explanation."""
        parts = []
        
        parts.append(f"# {explanation['title']}")
        parts.append("")
        parts.append(f"**Cycle:** `{explanation['cycle_id']}`")
        parts.append("")
        parts.append(f"## Summary")
        parts.append(explanation['summary'])
        parts.append("")
        
        stats = explanation.get('statistics', {})
        parts.append(f"**Statistics:** {stats.get('failed_files', 0)}/{stats.get('total_files', 0)} files failed, "
                    f"{stats.get('iterations_attempted', 0)} iteration(s) attempted")
        parts.append("")
        
        # Failure patterns
        if explanation.get('failure_patterns'):
            parts.append("## Detected Issues")
            for pattern in explanation['failure_patterns']:
                parts.append(f"\n### {pattern['pattern'].replace('_', ' ').title()}")
                parts.append(pattern['description'])
                if pattern['affected_files']:
                    parts.append(f"\n**Affected files:** {', '.join(f'`{f}`' for f in pattern['affected_files'][:3])}")
            parts.append("")
        
        # Recommendations
        if explanation.get('recommendations'):
            parts.append("## Recommendations")
            for rec in explanation['recommendations']:
                parts.append(f"\n### {rec['issue']}")
                parts.append(rec['suggestion'])
            parts.append("")
        
        # Per-file analysis
        if explanation.get('per_file_analysis'):
            parts.append("## Per-File Analysis")
            for analysis in explanation['per_file_analysis']:
                parts.append(f"\n### `{analysis['path']}` ({analysis['status']})")
                parts.append(f"**What was attempted:** {analysis['what_was_attempted']}")
                parts.append(f"**Why it failed:** {analysis['why_it_failed']}")
            parts.append("")
        
        # LLM intent
        parts.append("## LLM's Intended Approach")
        intent = explanation.get('llm_intent', {})
        if intent.get('strategy'):
            parts.append(f"**Strategy:** {intent['strategy']}")
        if intent.get('reasoning'):
            parts.append(f"**Reasoning:** {intent['reasoning']}")
        parts.append("")
        
        # The actual changes for manual application
        parts.append(explanation.get('llm_changes_for_manual_application', '(No changes available)'))
        parts.append("")
        
        # Next steps
        parts.append("## Next Steps")
        for step in explanation.get('next_steps', []):
            parts.append(step)
        parts.append("")
        
        return "\n".join(parts)
    
    # -------------------------------------------------------------------------
    # LLM-Enhanced Explanation (optional)
    # -------------------------------------------------------------------------
    
    def _build_llm_prompt(
        self,
        cycle: CycleSpec,
        description: Optional[CycleDescription],
        proposal: RefactorProposal,
        patterns: List[FailurePattern],
        file_analyses: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM to enhance failure explanation."""
        prompt = f"""You are helping a developer understand why an automated refactoring failed.

## Cycle Information
- ID: {cycle.id}
- Nodes: {', '.join(cycle.graph.nodes[:5])}

## Problem Description
{description.text[:800] if description and description.text else 'N/A'}

## What Failed
The LLM proposed patches that could not be applied. Here's what happened:

### Failure Patterns Detected
"""
        for pattern in patterns:
            prompt += f"- **{pattern.pattern_type}**: {pattern.likely_cause}\n"
        
        prompt += "\n### Files That Failed\n"
        for analysis in file_analyses[:3]:
            prompt += f"- `{analysis['path']}`: {analysis['why_it_failed']}\n"
        
        if proposal.llm_response:
            prompt += f"""
### LLM's Original Response (first 1500 chars)
```
{proposal.llm_response[:1500]}
```
"""
        
        prompt += """
## Your Task
Write a helpful explanation for a human developer that includes:
1. A clear summary of what went wrong
2. Specific, actionable suggestions for manual remediation
3. What the LLM was trying to accomplish (if discernible)

Output as JSON:
{
  "enhanced_summary": "<2-3 sentences explaining the situation>",
  "root_cause_analysis": "<what likely caused the failures>",
  "manual_remediation_steps": ["<step 1>", "<step 2>", ...],
  "alternative_approaches": ["<alternative 1>", ...]
}

Only output the JSON.
"""
        return prompt
    
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    def run(
        self,
        cycle_spec: Union[CycleSpec, Dict[str, Any]],
        description: Union[CycleDescription, Dict[str, Any], None] = None,
        proposal: Union[RefactorProposal, Dict[str, Any], None] = None,
        validation: Union[ValidationReport, Dict[str, Any], None] = None,
        validation_history: List[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Generate a failure explanation when patches cannot be applied.
        
        Args:
            cycle_spec: The cycle being refactored.
            description: Description from Describer agent.
            proposal: The failed RefactorProposal.
            validation: Final validation report (if any).
            validation_history: List of all validation attempts.
            
        Returns:
            AgentResult with failure explanation and remediation guidance.
        """
        logger.info("FailureExplainerAgent.run() starting")
        
        # Convert inputs to models
        if isinstance(cycle_spec, dict):
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        if description and isinstance(description, dict):
            description = CycleDescription.model_validate(description)
        if proposal is None:
            return AgentResult(
                status="error",
                output=None,
                logs="No proposal provided to explain",
            )
        if isinstance(proposal, dict):
            proposal = RefactorProposal.model_validate(proposal)
        if validation and isinstance(validation, dict):
            validation = ValidationReport.model_validate(validation)
        if validation_history is None:
            validation_history = []
        
        # Generate base explanation
        explanation = self._generate_failure_explanation(
            cycle_spec, description, proposal, validation, validation_history
        )
        
        # Optionally enhance with LLM
        if self.llm and self.include_llm_suggestions:
            try:
                patterns = self._detect_failure_patterns(proposal, validation)
                file_analyses = self._generate_per_file_analysis(proposal, patterns)
                prompt = self._build_llm_prompt(
                    cycle_spec, description, proposal, patterns, file_analyses
                )
                
                logger.info("Calling LLM for enhanced failure explanation")
                response = call_llm(self.llm, prompt)
                text = response if isinstance(response, str) else json.dumps(response)
                
                # Try to parse LLM response
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    enhanced = json.loads(json_match.group())
                    explanation["enhanced_summary"] = enhanced.get("enhanced_summary")
                    explanation["root_cause_analysis"] = enhanced.get("root_cause_analysis")
                    if enhanced.get("manual_remediation_steps"):
                        explanation["llm_remediation_steps"] = enhanced["manual_remediation_steps"]
                    if enhanced.get("alternative_approaches"):
                        explanation["alternative_approaches"] = enhanced["alternative_approaches"]
                    logger.info("LLM enhancement applied successfully")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(explanation)
        explanation["markdown_report"] = markdown_report
        
        logger.info(f"FailureExplainerAgent completed: {explanation['title']}")
        
        return AgentResult(
            status="success",
            output=explanation,
        )
