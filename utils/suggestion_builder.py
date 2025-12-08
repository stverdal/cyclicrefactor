"""Suggestion builder for human-reviewable refactoring suggestions.

This module builds suggestions from LLM output without trying to apply patches.
It provides rich context and explanations for human review.

This produces a ONE-STOP-SHOP SuggestionReport containing:
- Cycle context (what's the problem)
- LLM suggestions (what to do about it)
- Operator guidance (step-by-step instructions)
- Failure patterns (what might go wrong)
- Quick reference (files to create/modify, imports to change)
- Diagnostics (for troubleshooting)
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from models.schemas import (
    CycleSpec, 
    RefactorSuggestion, 
    CodeChange, 
    SuggestionReport,
    SuggestionValidation,
    CycleContext,
    OperatorGuidance,
    DiagnosticInfo,
    FailurePattern,
)
from utils.logging import get_logger

logger = get_logger("suggestion_builder")

# Default context lines to show before/after changes
DEFAULT_CONTEXT_LINES = 7


def build_suggestion_report(
    cycle_spec: CycleSpec,
    llm_response: str,
    strategy: Optional[str] = "",
    strategy_rationale: Optional[str] = "",
    context_lines: int = DEFAULT_CONTEXT_LINES,
    llm_model: str = "",
    llm_provider: str = "",
    generation_time_ms: int = 0,
) -> SuggestionReport:
    """Build a comprehensive SuggestionReport from LLM response.
    
    This builds a ONE-STOP-SHOP report containing everything an operator needs.
    
    Args:
        cycle_spec: The cycle specification with file contents (CycleSpec object or dict)
        llm_response: Raw LLM response (JSON or structured text)
        strategy: The refactoring strategy used (None will be converted to empty string)
        strategy_rationale: Why this strategy was chosen (None will be converted to empty string)
        context_lines: Number of context lines to show before/after
        llm_model: The LLM model used (for diagnostics)
        llm_provider: The LLM provider (for diagnostics)
        generation_time_ms: Time taken to generate response (for diagnostics)
        
    Returns:
        SuggestionReport ready for human review
    """
    # Handle None values - Pydantic requires str, not None
    strategy = strategy or ""
    strategy_rationale = strategy_rationale or ""
    
    # Handle dict input for backward compatibility
    if isinstance(cycle_spec, dict):
        try:
            cycle_spec = CycleSpec.model_validate(cycle_spec)
        except Exception as e:
            logger.warning(f"Failed to convert dict to CycleSpec: {e}")
            # Create a minimal CycleSpec
            from models.schemas import GraphSpec, FileSpec
            cycle_spec = CycleSpec(
                id=cycle_spec.get("id", "unknown"),
                graph=GraphSpec(nodes=[], edges=[]),
                files=[FileSpec(path=f.get("path", ""), content=f.get("content", "")) 
                       for f in cycle_spec.get("files", [])],
            )
    
    report = SuggestionReport(
        cycle_id=cycle_spec.id,
        strategy=strategy,
        strategy_rationale=strategy_rationale,
        llm_response=llm_response,
    )
    
    # Build cycle context
    report.cycle_context = _build_cycle_context(cycle_spec)
    
    # Build diagnostics
    report.diagnostics = DiagnosticInfo(
        llm_model=llm_model,
        llm_provider=llm_provider,
        generation_time_ms=generation_time_ms,
        timestamp=datetime.now().isoformat(),
    )
    
    # Parse LLM response
    parsed = _parse_llm_response(llm_response)
    if not parsed:
        report.warnings.append("Could not parse LLM response")
        report.failure_patterns.append(FailurePattern(
            pattern_type="parse_error",
            description="Could not parse LLM response as JSON",
            likely_cause="LLM may have returned malformed JSON or non-JSON text",
            remediation="Check the raw LLM response at the bottom of this report and manually extract the suggested changes",
        ))
        # Add basic operator guidance even on parse failure
        report.operator_guidance = _build_fallback_guidance(cycle_spec, llm_response)
        return report
    
    # Extract strategy if not provided
    if not report.strategy:
        report.strategy = parsed.get("strategy_used", parsed.get("strategy", "unknown"))
    
    # Extract reasoning as rationale
    if not report.strategy_rationale:
        report.strategy_rationale = parsed.get("reasoning", parsed.get("rationale", ""))
    
    # Build suggestions from new_files
    for new_file in parsed.get("new_files", []):
        suggestion = _build_new_file_suggestion(new_file, cycle_spec)
        if suggestion:
            report.suggestions.append(suggestion)
            report.suggested_order.append(suggestion.file_path)
            report.files_to_create.append(suggestion.file_path)
    
    # Build suggestions from changes/patches
    for change in parsed.get("changes", parsed.get("patches", [])):
        suggestion = _build_modification_suggestion(
            change, 
            cycle_spec, 
            context_lines
        )
        if suggestion:
            report.suggestions.append(suggestion)
            if suggestion.file_path not in report.suggested_order:
                report.suggested_order.append(suggestion.file_path)
            if suggestion.file_path not in report.files_to_modify:
                report.files_to_modify.append(suggestion.file_path)
            
            # Extract import changes
            _extract_import_changes(suggestion, report)
    
    # Calculate confidence based on completeness
    report.confidence = _calculate_confidence(report, parsed)
    
    # Build comprehensive operator guidance
    report.operator_guidance = _build_operator_guidance(report, cycle_spec, strategy)
    
    return report


def _build_cycle_context(cycle_spec: CycleSpec) -> CycleContext:
    """Build context about the cycle for operator understanding."""
    ctx = CycleContext(
        nodes=list(cycle_spec.graph.nodes) if cycle_spec.graph else [],
    )
    
    # Build edges
    if cycle_spec.graph and cycle_spec.graph.edges:
        ctx.edges = [
            {"source": e.source, "target": e.target}
            for e in cycle_spec.graph.edges
        ]
    
    # Determine cycle type
    node_count = len(ctx.nodes)
    if node_count == 2:
        ctx.cycle_type = "bidirectional"
        ctx.dependency_description = f"Files `{ctx.nodes[0]}` and `{ctx.nodes[1]}` depend on each other directly."
    elif node_count == 3:
        ctx.cycle_type = "triangle"
        ctx.dependency_description = f"A three-way dependency cycle exists between {', '.join(f'`{n}`' for n in ctx.nodes)}."
    else:
        ctx.cycle_type = "multi-node"
        ctx.dependency_description = f"A complex dependency cycle involving {node_count} files."
    
    # Find hotspot (most connected node)
    if ctx.edges:
        node_counts: Dict[str, int] = {}
        for edge in ctx.edges:
            node_counts[edge["source"]] = node_counts.get(edge["source"], 0) + 1
            node_counts[edge["target"]] = node_counts.get(edge["target"], 0) + 1
        if node_counts:
            ctx.hotspot_file = max(node_counts, key=node_counts.get)
    
    return ctx


def _build_fallback_guidance(cycle_spec: CycleSpec, llm_response: str) -> OperatorGuidance:
    """Build basic guidance when LLM response parsing fails."""
    return OperatorGuidance(
        executive_summary="The LLM response could not be parsed automatically. Manual review required.",
        difficulty_rating="complex",
        estimated_time="30-60 minutes",
        prerequisites=[
            "Review the raw LLM response below",
            "Identify the suggested changes manually",
            "Understand which imports create the cycle",
        ],
        step_by_step=[
            "Scroll to the 'Raw LLM Response' section at the bottom",
            "Look for suggested file changes (search for 'new_files' or 'changes')",
            "Identify which imports need to be added or removed",
            "Apply the changes manually in your IDE",
            "Verify the cycle is broken by running the cycle detector",
        ],
        verification_steps=[
            "Run the cycle detector to confirm the cycle is broken",
            "Build the project to ensure no compilation errors",
            "Run tests to verify no regressions",
        ],
        common_pitfalls=[
            "The LLM may have hallucinated file paths or type names - verify they exist",
            "Import paths may not be exact - adjust for your project structure",
        ],
        when_to_escalate="If the LLM response is completely garbled or empty, try re-running with a different model or more context.",
    )


def _build_operator_guidance(
    report: SuggestionReport,
    cycle_spec: CycleSpec,
    strategy: str,
) -> OperatorGuidance:
    """Build comprehensive operator guidance based on the suggestions."""
    
    # Determine difficulty based on number and type of changes
    new_files = len(report.files_to_create)
    mod_files = len(report.files_to_modify)
    total_changes = new_files + mod_files
    
    if total_changes <= 2 and new_files == 0:
        difficulty = "easy"
        time_est = "5-10 minutes"
    elif total_changes <= 4:
        difficulty = "moderate"
        time_est = "15-30 minutes"
    elif new_files > 0:
        difficulty = "moderate"
        time_est = "20-45 minutes"
    else:
        difficulty = "complex"
        time_est = "45-90 minutes"
    
    # Build step-by-step based on strategy
    steps = []
    verification = []
    pitfalls = []
    prereqs = []
    alternatives = []
    
    prereqs.append("Ensure you have a clean working directory (commit or stash changes)")
    prereqs.append("Have your IDE open with the project loaded")
    
    if report.files_to_create:
        steps.append(f"Create {new_files} new file(s): {', '.join(f'`{f}`' for f in report.files_to_create)}")
        pitfalls.append("Ensure new files are in the correct directory and have proper namespaces/packages")
    
    if report.files_to_modify:
        steps.append(f"Modify {mod_files} existing file(s): {', '.join(f'`{f}`' for f in report.files_to_modify)}")
    
    if report.imports_to_add:
        steps.append("Add the required import statements")
        pitfalls.append("Import paths may need adjustment for your project structure")
    
    if report.imports_to_remove:
        steps.append("Remove the old import statements that create the cycle")
    
    steps.append("Review each change carefully before applying")
    steps.append("Build/compile to check for errors")
    steps.append("Run tests to verify no regressions")
    
    verification = [
        "Build completes without errors",
        "Run the cycle detector - this cycle should no longer appear",
        "Unit tests pass",
        "The refactored code behavior is unchanged",
    ]
    
    # Strategy-specific guidance
    if "interface" in strategy.lower():
        steps.insert(1, "Create the interface file first (it has no dependencies)")
        pitfalls.append("Ensure all interface methods are implemented in the concrete class")
        alternatives.append("If interface extraction is too invasive, consider Dependency Inversion with abstract base class")
    elif "inversion" in strategy.lower():
        pitfalls.append("Ensure the abstraction is placed in a layer that both sides can access")
        alternatives.append("Consider using a shared module instead if abstractions feel forced")
    elif "shared" in strategy.lower() or "common" in strategy.lower():
        pitfalls.append("Don't move too much into the shared module - only what's needed to break the cycle")
        alternatives.append("Consider if the shared code belongs in a utility/common package")
    
    # Add common pitfalls
    pitfalls.extend([
        "Don't forget to update any namespace/package declarations in new files",
        "If using dependency injection, ensure the container is updated",
        "Watch for transitive dependencies that may recreate the cycle",
    ])
    
    # Build executive summary
    if report.cycle_will_be_broken:
        exec_summary = f"Apply {total_changes} change(s) using **{strategy or 'suggested'}** strategy to break this cycle."
    else:
        exec_summary = f"Review {total_changes} suggested change(s). Manual verification needed to confirm cycle is broken."
    
    return OperatorGuidance(
        executive_summary=exec_summary,
        difficulty_rating=difficulty,
        estimated_time=time_est,
        prerequisites=prereqs,
        step_by_step=steps,
        verification_steps=verification,
        common_pitfalls=pitfalls,
        when_to_escalate="If compilation errors persist after applying changes, or if the cycle reappears in a different form, consult with the architecture team.",
        alternative_approaches=alternatives,
    )


def _extract_import_changes(suggestion: RefactorSuggestion, report: SuggestionReport):
    """Extract import additions/removals from a suggestion."""
    for change in suggestion.changes:
        # Look for imports in suggested code that aren't in original
        orig_imports = _extract_imports(change.original_code)
        new_imports = _extract_imports(change.suggested_code)
        
        for imp in new_imports - orig_imports:
            report.imports_to_add.append({"file": suggestion.file_path, "import": imp})
        
        for imp in orig_imports - new_imports:
            report.imports_to_remove.append({"file": suggestion.file_path, "import": imp})


def _extract_imports(code: str) -> set:
    """Extract import statements from code."""
    imports = set()
    
    # Python: import X, from X import Y
    for match in re.finditer(r'^(?:from\s+\S+\s+)?import\s+.+$', code, re.MULTILINE):
        imports.add(match.group(0).strip())
    
    # C#: using X;
    for match in re.finditer(r'^using\s+[\w.]+;', code, re.MULTILINE):
        imports.add(match.group(0).strip())
    
    # Java: import X;
    for match in re.finditer(r'^import\s+[\w.*]+;', code, re.MULTILINE):
        imports.add(match.group(0).strip())
    
    # TypeScript/JS: import { X } from 'Y'
    for match in re.finditer(r'^import\s+.+\s+from\s+[\'"].+[\'"];?', code, re.MULTILINE):
        imports.add(match.group(0).strip())
    
    return imports


def _parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into structured data.
    
    Handles:
    - JSON responses
    - JSON wrapped in markdown code blocks
    - Partial/malformed JSON
    """
    if not response:
        return None
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        response = json_match.group(1)
    
    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in response
    json_start = response.find('{')
    json_end = response.rfind('}')
    if json_start >= 0 and json_end > json_start:
        try:
            return json.loads(response[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass
    
    logger.warning("Could not parse LLM response as JSON")
    return None


def _build_new_file_suggestion(new_file: Dict[str, Any], cycle_spec: CycleSpec = None) -> Optional[RefactorSuggestion]:
    """Build a suggestion for a new file with operator guidance."""
    path = new_file.get("path", "")
    content = new_file.get("content", "")
    
    if not path:
        return None
    
    # Determine purpose from path or content
    purpose = new_file.get("purpose", "")
    if not purpose:
        if "interface" in path.lower() or content.strip().startswith("public interface"):
            purpose = "interface"
        elif "abstract" in content.lower():
            purpose = "abstract class"
        else:
            purpose = "new module"
    
    # Build step-by-step manual instructions
    manual_steps = [
        f"Create a new file at path: `{path}`",
        "Copy the content below into the file",
        "Ensure namespace/package declaration matches your project structure",
        "Verify all imports are available in your project",
    ]
    
    # Common mistakes for new files
    common_mistakes = [
        "Wrong directory - ensure the file is in the correct folder",
        "Missing namespace/package - update to match your project conventions",
        "Incorrect imports - adjust import paths for your project structure",
    ]
    
    # Testing guidance
    if "interface" in purpose.lower():
        testing_notes = "Verify the interface compiles. Implementers will be updated in subsequent changes."
    elif "abstract" in purpose.lower():
        testing_notes = "Verify the abstract class compiles and defines all necessary abstract members."
    else:
        testing_notes = "Verify the new file compiles and has no missing dependencies."
    
    return RefactorSuggestion(
        file_path=path,
        title=f"Create {purpose}",
        explanation=f"Create new file `{path}` containing {purpose} to break the dependency cycle.",
        is_new_file=True,
        new_file_content=content,
        copy_paste_ready=content,  # Same as content for new files
        confidence=0.9 if content else 0.5,
        validation_notes=[
            f"Purpose: {purpose}",
            f"Content length: {len(content)} characters",
        ],
        manual_steps=manual_steps,
        common_mistakes=common_mistakes,
        testing_notes=testing_notes,
        rollback_instructions=f"Simply delete the file `{path}` to undo this change.",
        prerequisites=["Ensure the target directory exists"],
    )


def _build_modification_suggestion(
    change: Dict[str, Any],
    cycle_spec: CycleSpec,
    context_lines: int,
) -> Optional[RefactorSuggestion]:
    """Build a suggestion for modifying an existing file."""
    path = change.get("path", "")
    if not path:
        return None
    
    # Get original file content
    original_content = cycle_spec.get_file_content(path) or ""
    original_lines = original_content.splitlines() if original_content else []
    
    suggestion = RefactorSuggestion(
        file_path=path,
        title=f"Modify {path.split('/')[-1].split('\\')[-1]}",
        explanation="",
        is_new_file=False,
        changes=[],
        validation_notes=[],
    )
    
    explanations = []
    
    # Handle prepend (new imports)
    prepend = change.get("prepend", "")
    if prepend:
        suggestion.changes.append(CodeChange(
            line_start=1,
            line_end=1,
            original_code="(beginning of file)",
            suggested_code=prepend,
            change_type="add",
        ))
        explanations.append(f"Add import: `{prepend.strip()}`")
    
    # Handle search_replace blocks
    search_replace = change.get("search_replace", [])
    for sr in search_replace:
        search = sr.get("search", "")
        replace = sr.get("replace", "")
        
        if not search:
            continue
        
        # Find the search text in original content
        line_start, line_end = _find_text_location(original_content, search)
        
        # Get context
        context_before = ""
        context_after = ""
        if original_lines and line_start > 0:
            start_ctx = max(0, line_start - 1 - context_lines)
            end_ctx = min(len(original_lines), line_end + context_lines)
            context_before = "\n".join(original_lines[start_ctx:line_start - 1])
            context_after = "\n".join(original_lines[line_end:end_ctx])
        
        code_change = CodeChange(
            line_start=line_start,
            line_end=line_end,
            original_code=search,
            suggested_code=replace,
            change_type="modify" if replace else "remove",
        )
        suggestion.changes.append(code_change)
        
        # Store context
        if context_before and not suggestion.context_before:
            suggestion.context_before = context_before
        if context_after:
            suggestion.context_after = context_after
        
        # Generate explanation for this change
        change_desc = _describe_change(search, replace)
        if change_desc:
            explanations.append(change_desc)
    
    # Handle patched (full file replacement) - less ideal but handle it
    if "patched" in change and not search_replace:
        patched = change.get("patched", "")
        suggestion.changes.append(CodeChange(
            line_start=1,
            line_end=len(original_lines) if original_lines else 1,
            original_code="(full file - see original)",
            suggested_code=patched,
            change_type="modify",
        ))
        explanations.append("Replace entire file content")
        suggestion.validation_notes.append("⚠️ Full file replacement - review carefully")
    
    # Build explanation
    suggestion.explanation = "; ".join(explanations) if explanations else "Modify file to break cycle"
    
    # Calculate confidence
    if not suggestion.changes:
        suggestion.confidence = 0.3
        suggestion.validation_notes.append("⚠️ No specific changes identified")
    elif any(c.line_start == 0 for c in suggestion.changes):
        suggestion.confidence = 0.6
        suggestion.validation_notes.append("⚠️ Could not locate exact position of some changes")
    else:
        suggestion.confidence = 0.85
    
    # Add operator guidance for modifications
    suggestion.manual_steps = [
        f"Open file: `{path}`",
    ]
    
    for i, chg in enumerate(suggestion.changes, 1):
        if chg.change_type == "add":
            suggestion.manual_steps.append(f"Add the code from Change {i} at the specified location")
        elif chg.change_type == "remove":
            suggestion.manual_steps.append(f"Remove the code shown in Change {i}")
        else:
            if chg.line_start > 0:
                suggestion.manual_steps.append(f"Go to line {chg.line_start} and replace the code as shown in Change {i}")
            else:
                suggestion.manual_steps.append(f"Find and replace the code as shown in Change {i} (use search)")
    
    suggestion.manual_steps.append("Save the file")
    suggestion.manual_steps.append("Verify the file compiles without errors")
    
    suggestion.testing_notes = "After making this change, verify the file compiles and any dependent code still works."
    
    # Common mistakes for modifications
    suggestion.common_mistakes = [
        "Search text may have whitespace differences - use your IDE's search if exact match fails",
        "Line numbers may shift if you've made other changes - search for the code instead",
    ]
    
    if any(c.line_start == 0 for c in suggestion.changes):
        suggestion.common_mistakes.append(
            "Some changes couldn't be precisely located - manually search for the 'Find this code' text"
        )
    
    suggestion.rollback_instructions = f"Use version control (git checkout `{path}`) or undo in your editor."
    
    # Add potential issues to each change
    for chg in suggestion.changes:
        if chg.line_start == 0:
            chg.potential_issues.append("Could not locate exact line - search for this code manually")
        if chg.change_type == "modify" and len(chg.suggested_code) > len(chg.original_code) * 2:
            chg.potential_issues.append("Significant code expansion - review carefully for correctness")
        if "import" in chg.suggested_code.lower() or "using" in chg.suggested_code.lower():
            chg.why_needed = "Update imports to reference the new abstraction instead of concrete implementation"
    
    return suggestion


def _find_text_location(content: str, search_text: str) -> Tuple[int, int]:
    """Find the line numbers where search_text appears in content.
    
    Returns:
        Tuple of (start_line, end_line) - 1-indexed. Returns (0, 0) if not found.
    """
    if not content or not search_text:
        return (0, 0)
    
    # Normalize whitespace for matching
    normalized_content = content
    normalized_search = search_text
    
    # Try exact match first
    pos = content.find(search_text)
    if pos < 0:
        # Try with normalized whitespace
        normalized_content = re.sub(r'[ \t]+', ' ', content)
        normalized_search = re.sub(r'[ \t]+', ' ', search_text)
        pos = normalized_content.find(normalized_search)
    
    if pos < 0:
        return (0, 0)
    
    # Count lines to find position
    lines_before = content[:pos].count('\n')
    search_lines = search_text.count('\n')
    
    start_line = lines_before + 1
    end_line = start_line + search_lines
    
    return (start_line, end_line)


def _describe_change(original: str, replacement: str) -> str:
    """Generate a human-readable description of a change."""
    if not original and replacement:
        return "Add new code"
    if original and not replacement:
        return "Remove code"
    if original == replacement:
        return ""  # No-op
    
    # Detect common patterns
    if "interface" in replacement.lower() and "interface" not in original.lower():
        return "Add interface implementation"
    
    if "using " in replacement or "import " in replacement:
        if "using " not in original and "import " not in original:
            return "Add import statement"
        else:
            return "Change import statement"
    
    # Look for type changes
    type_pattern = r':\s*(\w+)'
    orig_types = re.findall(type_pattern, original)
    new_types = re.findall(type_pattern, replacement)
    if orig_types and new_types and orig_types != new_types:
        return f"Change type from `{orig_types[0]}` to `{new_types[0]}`"
    
    # Generic description based on size
    orig_lines = len(original.splitlines())
    new_lines = len(replacement.splitlines())
    if new_lines > orig_lines:
        return f"Expand code ({orig_lines} → {new_lines} lines)"
    elif new_lines < orig_lines:
        return f"Simplify code ({orig_lines} → {new_lines} lines)"
    else:
        return "Modify code"


def _calculate_confidence(report: SuggestionReport, parsed: Dict[str, Any]) -> float:
    """Calculate overall confidence score for the suggestions."""
    if not report.suggestions:
        return 0.0
    
    # Average of individual suggestion confidences
    avg_confidence = sum(s.confidence for s in report.suggestions) / len(report.suggestions)
    
    # Bonus for having new files (scaffolding)
    has_new_files = any(s.is_new_file for s in report.suggestions)
    if has_new_files:
        avg_confidence += 0.05
    
    # Penalty for warnings
    warning_penalty = len(report.warnings) * 0.1
    avg_confidence -= warning_penalty
    
    # Clamp to 0-1
    return max(0.0, min(1.0, avg_confidence))


def enrich_with_context(
    suggestions: List[RefactorSuggestion],
    cycle_spec: CycleSpec,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> List[RefactorSuggestion]:
    """Enrich suggestions with additional context from source files.
    
    Args:
        suggestions: List of suggestions to enrich
        cycle_spec: Cycle spec with file contents
        context_lines: Number of context lines to add
        
    Returns:
        Enriched suggestions
    """
    for suggestion in suggestions:
        if suggestion.is_new_file:
            continue
        
        content = cycle_spec.get_file_content(suggestion.file_path)
        if not content:
            continue
        
        lines = content.splitlines()
        
        for change in suggestion.changes:
            if change.line_start > 0 and change.line_end > 0:
                # Add context before
                start_ctx = max(0, change.line_start - 1 - context_lines)
                end_ctx = min(len(lines), change.line_end + context_lines)
                
                if not suggestion.context_before:
                    suggestion.context_before = "\n".join(lines[start_ctx:change.line_start - 1])
                if not suggestion.context_after:
                    suggestion.context_after = "\n".join(lines[change.line_end:end_ctx])
    
    return suggestions
