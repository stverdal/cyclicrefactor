"""Suggestion builder for human-reviewable refactoring suggestions.

This module builds suggestions from LLM output without trying to apply patches.
It provides rich context and explanations for human review.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from models.schemas import (
    CycleSpec, 
    RefactorSuggestion, 
    CodeChange, 
    SuggestionReport,
    SuggestionValidation,
)
from utils.logging import get_logger

logger = get_logger("suggestion_builder")

# Default context lines to show before/after changes
DEFAULT_CONTEXT_LINES = 7


def build_suggestion_report(
    cycle_spec: CycleSpec,
    llm_response: str,
    strategy: str = "",
    strategy_rationale: str = "",
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> SuggestionReport:
    """Build a SuggestionReport from LLM response.
    
    Args:
        cycle_spec: The cycle specification with file contents (CycleSpec object or dict)
        llm_response: Raw LLM response (JSON or structured text)
        strategy: The refactoring strategy used
        strategy_rationale: Why this strategy was chosen
        context_lines: Number of context lines to show before/after
        
    Returns:
        SuggestionReport ready for human review
    """
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
    
    # Parse LLM response
    parsed = _parse_llm_response(llm_response)
    if not parsed:
        report.warnings.append("Could not parse LLM response")
        return report
    
    # Extract strategy if not provided
    if not report.strategy:
        report.strategy = parsed.get("strategy_used", parsed.get("strategy", "unknown"))
    
    # Extract reasoning as rationale
    if not report.strategy_rationale:
        report.strategy_rationale = parsed.get("reasoning", parsed.get("rationale", ""))
    
    # Build suggestions from new_files
    for new_file in parsed.get("new_files", []):
        suggestion = _build_new_file_suggestion(new_file)
        if suggestion:
            report.suggestions.append(suggestion)
            report.suggested_order.append(suggestion.file_path)
    
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
    
    # Calculate confidence based on completeness
    report.confidence = _calculate_confidence(report, parsed)
    
    return report


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


def _build_new_file_suggestion(new_file: Dict[str, Any]) -> Optional[RefactorSuggestion]:
    """Build a suggestion for a new file."""
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
    
    return RefactorSuggestion(
        file_path=path,
        title=f"Create {purpose}",
        explanation=f"Create new file `{path}` containing {purpose} to break the dependency cycle.",
        is_new_file=True,
        new_file_content=content,
        confidence=0.9 if content else 0.5,
        validation_notes=[
            f"Purpose: {purpose}",
            f"Content length: {len(content)} characters",
        ],
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
