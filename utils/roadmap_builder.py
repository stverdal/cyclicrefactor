"""Refactor Roadmap builder for demo-friendly output.

This module builds a RefactorRoadmap that provides visibility into:
- What was attempted (even if it failed)
- What succeeded
- What remains to be done
- Classification of failures (hallucination, syntax, etc.)

This is particularly useful for demonstrations where you want to show
the pipeline's capabilities even when full automation isn't achieved.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from models.schemas import (
    RefactorRoadmap, 
    Patch, 
    PartialAttempt, 
    FailureClassification,
    ScaffoldFile,
)
from utils.logging import get_logger

logger = get_logger("roadmap_builder")


class RoadmapBuilder:
    """Builder class for constructing RefactorRoadmap incrementally."""
    
    def __init__(self, cycle_id: str):
        self.roadmap = RefactorRoadmap(cycle_id=cycle_id)
        self.llm_call_count = 0
        self.start_time = datetime.now()
    
    def set_strategy(self, strategy: str, rationale: str = "", confidence: float = 0.0):
        """Set the chosen strategy."""
        self.roadmap.strategy_chosen = strategy
        self.roadmap.strategy_rationale = rationale
        self.roadmap.confidence = confidence
        logger.debug(f"Roadmap: strategy={strategy}, confidence={confidence:.0%}")
    
    def add_llm_response(self, response: str):
        """Track LLM responses for debugging."""
        self.roadmap.llm_responses.append(response[:2000])  # Truncate for storage
        self.llm_call_count += 1
        self.roadmap.total_llm_calls = self.llm_call_count
    
    def add_scaffold_file(
        self, 
        path: str, 
        content: str, 
        purpose: str,
        validated: bool = False,
        errors: List[str] = None
    ):
        """Add a scaffold file result."""
        sf = ScaffoldFile(
            path=path,
            content=content,
            purpose=purpose,
            validated=validated,
            validation_errors=errors or []
        )
        self.roadmap.scaffold_files.append(sf)
        
        if not validated and errors:
            self.roadmap.scaffold_success = False
            logger.warning(f"Scaffold failed: {path} - {errors}")
    
    def add_successful_patch(self, patch: Patch):
        """Add a successfully applied patch."""
        self.roadmap.successful_patches.append(patch)
        logger.info(f"Roadmap: successful patch for {patch.path}")
    
    def add_partial_attempt(
        self,
        file_path: str,
        intended_changes: List[str],
        actual_changes: List[str],
        failure_reason: str,
        failure_category: str = "unknown",
        evidence: List[str] = None,
        raw_output: str = None,
        suggested_fix: str = ""
    ):
        """Add a partial or failed attempt."""
        classification = FailureClassification(
            category=failure_category,
            description=failure_reason,
            evidence=evidence or [],
            recoverable=failure_category not in ["hallucination", "fundamental_mismatch"]
        )
        
        attempt = PartialAttempt(
            file_path=file_path,
            intended_changes=intended_changes,
            actual_changes=actual_changes,
            failure_reason=failure_reason,
            failure_classification=classification,
            raw_llm_output=raw_output[:1000] if raw_output else None,
            suggested_fix=suggested_fix
        )
        self.roadmap.partial_attempts.append(attempt)
        logger.info(f"Roadmap: partial attempt for {file_path} - {failure_category}")
    
    def add_remaining_work(self, description: str):
        """Add a human-actionable remaining work item."""
        self.roadmap.remaining_work.append(description)
    
    def set_minimal_diff(self, target: str, patch: Optional[Patch]):
        """Set minimal diff recommendation."""
        self.roadmap.minimal_diff_target = target
        self.roadmap.minimal_diff_patch = patch
    
    def build(self) -> RefactorRoadmap:
        """Build and return the final roadmap."""
        # Calculate overall confidence
        if self.roadmap.successful_patches:
            success_rate = len(self.roadmap.successful_patches) / (
                len(self.roadmap.successful_patches) + len(self.roadmap.partial_attempts)
            )
            self.roadmap.confidence = max(self.roadmap.confidence, success_rate * 0.8)
        
        # Generate executive summary
        self.roadmap.executive_summary = self.roadmap.generate_executive_summary()
        
        logger.info(f"Roadmap built: {len(self.roadmap.successful_patches)} successes, "
                   f"{len(self.roadmap.partial_attempts)} partial, "
                   f"{len(self.roadmap.remaining_work)} remaining items")
        
        return self.roadmap


def classify_failure(
    patch_result: Dict[str, Any],
    original_content: str,
    llm_response: str,
) -> FailureClassification:
    """Classify why a patch failed.
    
    Categories:
    - hallucination: LLM invented types/methods that don't exist
    - syntax: Syntax errors in generated code
    - no_op: Search == replace (no actual change)
    - search_mismatch: Search text not found in file
    - validation: Failed rule-based validation
    - compile: Failed compile/lint check
    
    Args:
        patch_result: The patch processing result
        original_content: Original file content
        llm_response: Raw LLM response
        
    Returns:
        FailureClassification with category and evidence
    """
    evidence = []
    
    # Check for no-op
    sr_list = patch_result.get("search_replace", [])
    for sr in sr_list:
        if sr.get("search") == sr.get("replace"):
            return FailureClassification(
                category="no_op",
                description="Patch makes no actual change (search equals replace)",
                evidence=[f"search: {sr.get('search', '')[:100]}..."],
                recoverable=True
            )
    
    # Check for search mismatch
    warnings = patch_result.get("warnings", [])
    for w in warnings:
        if "not found" in w.lower() or "no match" in w.lower():
            evidence.append(w)
    if evidence:
        return FailureClassification(
            category="search_mismatch",
            description="Search text not found in file",
            evidence=evidence,
            recoverable=True
        )
    
    # Check for hallucinated types
    hallucination_patterns = [
        r'\bI[A-Z]\w+Service\b',  # IXxxService patterns
        r'\bI[A-Z]\w+Repository\b',
        r'\bI[A-Z]\w+Handler\b',
    ]
    
    for sr in sr_list:
        replace_text = sr.get("replace", "")
        for pattern in hallucination_patterns:
            matches = re.findall(pattern, replace_text)
            for m in matches:
                # Check if this type exists in original content or is defined in replacement
                if m not in original_content and f"interface {m}" not in llm_response:
                    evidence.append(f"Type '{m}' used but not defined")
    
    if evidence:
        return FailureClassification(
            category="hallucination",
            description="LLM introduced types that don't exist",
            evidence=evidence,
            recoverable=False
        )
    
    # Check for syntax errors in validation
    for w in warnings:
        if "syntax" in w.lower() or "bracket" in w.lower() or "brace" in w.lower():
            evidence.append(w)
    if evidence:
        return FailureClassification(
            category="syntax",
            description="Syntax errors in generated code",
            evidence=evidence,
            recoverable=True
        )
    
    # Check for validation failures
    validation_issues = patch_result.get("validation_issues", [])
    if validation_issues:
        return FailureClassification(
            category="validation",
            description="Failed rule-based validation",
            evidence=validation_issues[:5],
            recoverable=True
        )
    
    # Default
    return FailureClassification(
        category="unknown",
        description="Failure reason unclear",
        evidence=warnings[:5] if warnings else [],
        recoverable=True
    )


def generate_remaining_work(
    roadmap: RefactorRoadmap,
    cycle_spec: Dict[str, Any],
) -> List[str]:
    """Generate human-actionable remaining work items.
    
    Args:
        roadmap: Current roadmap state
        cycle_spec: Original cycle specification
        
    Returns:
        List of action items for humans
    """
    work_items = []
    
    # Check scaffolding failures
    for sf in roadmap.scaffold_files:
        if not sf.validated:
            work_items.append(
                f"Fix scaffold file `{sf.path}`: {', '.join(sf.validation_errors[:2])}"
            )
    
    # Check partial attempts
    for attempt in roadmap.partial_attempts:
        if attempt.suggested_fix:
            work_items.append(attempt.suggested_fix)
        elif attempt.failure_classification:
            cat = attempt.failure_classification.category
            if cat == "hallucination":
                work_items.append(
                    f"Create missing interface in `{attempt.file_path}` - "
                    f"LLM assumed it exists: {', '.join(attempt.failure_classification.evidence[:2])}"
                )
            elif cat == "search_mismatch":
                work_items.append(
                    f"Manually locate and modify code in `{attempt.file_path}` - "
                    f"automated search failed"
                )
            elif cat == "syntax":
                work_items.append(
                    f"Fix syntax errors in `{attempt.file_path}`: "
                    f"{', '.join(attempt.failure_classification.evidence[:2])}"
                )
    
    # Add generic guidance if strategy is clear but incomplete
    if roadmap.strategy_chosen and not roadmap.successful_patches:
        strategy_guidance = {
            "interface_extraction": [
                "Create an interface with methods needed by the dependent class",
                "Have the depended-on class implement the interface",
                "Update the dependent class to use the interface instead",
            ],
            "dependency_inversion": [
                "Create an abstraction in the higher-level module",
                "Have the lower-level module implement the abstraction",
                "Inject the implementation at runtime",
            ],
            "shared_module": [
                "Identify the common functionality used by both modules",
                "Extract it to a new shared module",
                "Update both modules to import from the shared module",
            ],
        }
        
        guidance = strategy_guidance.get(roadmap.strategy_chosen, [])
        for step in guidance:
            if step not in work_items:
                work_items.append(step)
    
    return work_items


def build_roadmap_from_results(
    cycle_id: str,
    strategy: str,
    patch_results: List[Dict[str, Any]],
    scaffold_results: Optional[Dict[str, Any]] = None,
    minimal_diff_result: Optional[Dict[str, Any]] = None,
    llm_responses: List[str] = None,
    cycle_spec: Dict[str, Any] = None,
) -> RefactorRoadmap:
    """Build a roadmap from refactoring results.
    
    This is the main entry point for converting refactoring results
    into a demo-friendly roadmap.
    
    Args:
        cycle_id: Cycle identifier
        strategy: Strategy used
        patch_results: List of patch processing results
        scaffold_results: Results from scaffolding phase (if used)
        minimal_diff_result: Result from minimal diff mode (if used)
        llm_responses: Raw LLM responses for debugging
        cycle_spec: Original cycle spec for context
        
    Returns:
        RefactorRoadmap
    """
    builder = RoadmapBuilder(cycle_id)
    builder.set_strategy(strategy)
    
    # Add LLM responses
    for resp in (llm_responses or []):
        builder.add_llm_response(resp)
    
    # Process scaffold results
    if scaffold_results:
        for sf in scaffold_results.get("files_created", []):
            builder.add_scaffold_file(
                path=sf.get("path", ""),
                content=sf.get("content", ""),
                purpose=sf.get("purpose", ""),
                validated=sf.get("valid", False),
                errors=sf.get("errors", [])
            )
    
    # Process patch results
    for pr in patch_results:
        path = pr.get("path", "")
        status = pr.get("status", "")
        
        if status in ["applied", "partial"] and pr.get("patched") != pr.get("original"):
            # Successful patch
            patch = Patch(
                path=path,
                original=pr.get("original", ""),
                patched=pr.get("patched", ""),
                diff=pr.get("diff", ""),
                status=status,
                warnings=pr.get("warnings", []),
                confidence=pr.get("confidence", 0.0),
            )
            builder.add_successful_patch(patch)
        else:
            # Failed or partial attempt
            classification = classify_failure(
                pr,
                pr.get("original", ""),
                llm_responses[0] if llm_responses else ""
            )
            
            builder.add_partial_attempt(
                file_path=path,
                intended_changes=pr.get("changes_made", []),
                actual_changes=[],
                failure_reason=pr.get("revert_reason", "") or classification.description,
                failure_category=classification.category,
                evidence=classification.evidence,
                suggested_fix=_generate_suggested_fix(classification, path)
            )
    
    # Add minimal diff info
    if minimal_diff_result:
        target = minimal_diff_result.get("target_edge", {})
        target_str = f"{target.get('from', '')} → {target.get('to', '')}"
        
        patch_data = minimal_diff_result.get("patch")
        patch = None
        if patch_data:
            patch = Patch(
                path=patch_data.get("path", ""),
                patched="",  # Would be filled in after application
                status="recommended",
            )
        
        builder.set_minimal_diff(target_str, patch)
    
    # Generate remaining work
    roadmap = builder.build()
    roadmap.remaining_work = generate_remaining_work(roadmap, cycle_spec or {})
    roadmap.executive_summary = roadmap.generate_executive_summary()
    
    return roadmap


def _generate_suggested_fix(classification: FailureClassification, path: str) -> str:
    """Generate a suggested fix based on failure classification."""
    suggestions = {
        "hallucination": f"Create the missing interface/type before modifying {path}",
        "syntax": f"Fix syntax errors in generated code for {path}",
        "search_mismatch": f"Manually locate the code block in {path} and apply the change",
        "no_op": "Regenerate the patch with clearer instructions",
        "validation": f"Review validation errors and adjust the approach for {path}",
        "compile": f"Fix compile errors in {path}",
    }
    return suggestions.get(classification.category, f"Review and fix issues in {path}")
