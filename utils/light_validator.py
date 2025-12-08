"""Light validator for suggestion mode.

This module performs semantic validation of refactoring suggestions without
strict syntax checking. It focuses on:
- Logical cycle breaking (does the change actually break the dependency?)
- Type existence (are referenced types defined somewhere?)
- Hallucination detection (did the LLM invent non-existent code?)

No compile checks or strict syntax validation - just logical checks.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from models.schemas import (
    CycleSpec,
    SuggestionReport,
    RefactorSuggestion,
    SuggestionValidation,
)
from utils.logging import get_logger

logger = get_logger("light_validator")


def validate_suggestions(
    report: SuggestionReport,
    cycle_spec: CycleSpec,
) -> SuggestionValidation:
    """Perform light validation on suggestions.
    
    Args:
        report: The suggestion report to validate
        cycle_spec: The original cycle specification
        
    Returns:
        SuggestionValidation with results
    """
    validation = SuggestionValidation()
    
    # Collect all types defined in original files
    original_types = _extract_defined_types(cycle_spec)
    
    # Collect types defined in new file suggestions
    suggested_types = _extract_suggested_types(report.suggestions)
    
    # All available types
    all_types = original_types | suggested_types
    
    # Check 1: All referenced types exist
    referenced_types = _extract_referenced_types(report.suggestions)
    missing_types = referenced_types - all_types
    
    if missing_types:
        validation.types_exist = False
        for t in missing_types:
            validation.warnings.append(f"Type '{t}' is referenced but not defined anywhere")
    
    # Check 2: Cycle logically broken
    cycle_broken, cycle_reason = _check_cycle_broken(report, cycle_spec)
    validation.cycle_logically_broken = cycle_broken
    if not cycle_broken:
        validation.warnings.append(f"Cycle may not be broken: {cycle_reason}")
    
    # Check 3: Hallucination detection
    hallucinations = _detect_hallucinations(report, cycle_spec)
    if hallucinations:
        validation.no_hallucinations_detected = False
        for h in hallucinations:
            validation.errors.append(f"Possible hallucination: {h}")
    
    # Check 4: Search text exists in original files
    search_issues = _validate_search_text(report.suggestions, cycle_spec)
    for issue in search_issues:
        validation.warnings.append(issue)
    
    # Check 5: No-op detection
    noops = _detect_noops(report.suggestions)
    for noop in noops:
        validation.warnings.append(noop)
    
    # Determine overall validity
    validation.is_valid = (
        len(validation.errors) == 0 and
        validation.types_exist and
        validation.no_hallucinations_detected
    )
    
    logger.info(f"Light validation: valid={validation.is_valid}, "
                f"cycle_broken={validation.cycle_logically_broken}, "
                f"warnings={len(validation.warnings)}, errors={len(validation.errors)}")
    
    return validation


def _extract_defined_types(cycle_spec: CycleSpec) -> Set[str]:
    """Extract all type names defined in original source files."""
    types = set()
    
    for file in cycle_spec.files:
        content = file.content or ""
        
        # Extract class/interface/type definitions
        # C# / Java style
        for match in re.finditer(r'\b(?:class|interface|struct|enum)\s+(\w+)', content):
            types.add(match.group(1))
        
        # TypeScript style
        for match in re.finditer(r'\b(?:type|interface)\s+(\w+)', content):
            types.add(match.group(1))
        
        # Python class
        for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            types.add(match.group(1))
    
    return types


def _extract_suggested_types(suggestions: List[RefactorSuggestion]) -> Set[str]:
    """Extract type names defined in new file suggestions."""
    types = set()
    
    for suggestion in suggestions:
        if not suggestion.is_new_file:
            continue
        
        content = suggestion.new_file_content
        
        # Extract definitions (same patterns as above)
        for match in re.finditer(r'\b(?:class|interface|struct|enum|type)\s+(\w+)', content):
            types.add(match.group(1))
        
        # Python class
        for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            types.add(match.group(1))
    
    return types


def _extract_referenced_types(suggestions: List[RefactorSuggestion]) -> Set[str]:
    """Extract type names referenced in suggested changes."""
    types = set()
    
    # Common built-in types to exclude
    builtin_types = {
        'string', 'int', 'float', 'double', 'bool', 'boolean', 'void', 'null',
        'object', 'any', 'number', 'String', 'Int', 'Float', 'Double', 'Boolean',
        'Object', 'Array', 'List', 'Dict', 'Set', 'Map', 'Promise', 'Task',
        'Optional', 'None', 'True', 'False', 'undefined',
    }
    
    for suggestion in suggestions:
        # Check changes
        for change in suggestion.changes:
            code = change.suggested_code
            
            # Look for type references in common patterns
            # : TypeName (type annotations)
            for match in re.finditer(r':\s*([A-Z]\w+)', code):
                t = match.group(1)
                if t not in builtin_types:
                    types.add(t)
            
            # new TypeName( (instantiation)
            for match in re.finditer(r'\bnew\s+([A-Z]\w+)', code):
                t = match.group(1)
                if t not in builtin_types:
                    types.add(t)
            
            # implements/extends TypeName
            for match in re.finditer(r'\b(?:implements|extends)\s+([A-Z]\w+)', code):
                types.add(match.group(1))
    
    return types


def _check_cycle_broken(
    report: SuggestionReport,
    cycle_spec: CycleSpec,
) -> Tuple[bool, str]:
    """Check if the suggested changes would logically break the cycle.
    
    Returns:
        Tuple of (is_broken, reason_if_not)
    """
    # Get the edges in the cycle
    edges = cycle_spec.graph.edges
    if not edges:
        return (False, "No edges in cycle graph")
    
    # Check if suggestions involve interface extraction
    has_new_interface = any(
        s.is_new_file and 'interface' in s.file_path.lower()
        for s in report.suggestions
    )
    
    # Check if there's an import change
    has_import_change = False
    for suggestion in report.suggestions:
        for change in suggestion.changes:
            if 'import' in change.suggested_code.lower() or 'using' in change.suggested_code.lower():
                has_import_change = True
                break
    
    # Check if dependency direction is inverted (references to interface instead of concrete)
    has_type_change = False
    for suggestion in report.suggestions:
        for change in suggestion.changes:
            # Look for I prefix pattern (IFoo instead of Foo)
            if re.search(r': I[A-Z]\w+', change.suggested_code):
                has_type_change = True
                break
    
    # Heuristic: if we have new interface + import change + type change, likely broken
    if has_new_interface and has_import_change:
        return (True, "")
    
    if has_new_interface and has_type_change:
        return (True, "")
    
    # Check for dependency removal patterns
    for suggestion in report.suggestions:
        for change in suggestion.changes:
            orig = change.original_code.lower()
            new = change.suggested_code.lower()
            
            # Import removed
            if 'import' in orig and 'import' not in new:
                return (True, "")
            if 'using' in orig and 'using' not in new:
                return (True, "")
            
            # Direct class reference replaced with interface
            if re.search(r'\b[A-Z]\w+\b', orig) and re.search(r'\bI[A-Z]\w+\b', new):
                return (True, "")
    
    # If strategy is parameter injection or lazy import, check for those patterns
    if report.strategy in ['parameter_injection', 'lazy_import']:
        for suggestion in report.suggestions:
            for change in suggestion.changes:
                # Parameter added to function signature
                if '(' in change.original_code and '(' in change.suggested_code:
                    orig_params = change.original_code.count(',')
                    new_params = change.suggested_code.count(',')
                    if new_params > orig_params:
                        return (True, "")
    
    return (False, "Could not confirm cycle is broken - manual verification needed")


def _detect_hallucinations(
    report: SuggestionReport,
    cycle_spec: CycleSpec,
) -> List[str]:
    """Detect potential hallucinations in suggestions.
    
    Returns:
        List of hallucination descriptions
    """
    hallucinations = []
    
    for suggestion in report.suggestions:
        if suggestion.is_new_file:
            continue
        
        # Get original file content
        original = cycle_spec.get_file_content(suggestion.file_path)
        if not original:
            hallucinations.append(f"File '{suggestion.file_path}' not found in cycle")
            continue
        
        for change in suggestion.changes:
            search = change.original_code
            
            # Skip if it's a pure add or metadata
            if not search or search == "(beginning of file)" or search == "(full file - see original)":
                continue
            
            # Check if search text exists in original
            if search not in original:
                # Try normalized match
                normalized_original = re.sub(r'\s+', ' ', original)
                normalized_search = re.sub(r'\s+', ' ', search)
                
                if normalized_search not in normalized_original:
                    # Truncate for readability
                    search_preview = search[:100] + "..." if len(search) > 100 else search
                    hallucinations.append(
                        f"Search text not found in {suggestion.file_path}: '{search_preview}'"
                    )
    
    return hallucinations


def _validate_search_text(
    suggestions: List[RefactorSuggestion],
    cycle_spec: CycleSpec,
) -> List[str]:
    """Validate that search text in changes exists in source files.
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    for suggestion in suggestions:
        if suggestion.is_new_file:
            continue
        
        original = cycle_spec.get_file_content(suggestion.file_path) or ""
        
        for change in suggestion.changes:
            if change.line_start == 0 and change.original_code:
                # Line position unknown - couldn't find the text
                warnings.append(
                    f"Could not locate exact position in {suggestion.file_path} - "
                    f"search text may not match exactly"
                )
    
    return warnings


def _detect_noops(suggestions: List[RefactorSuggestion]) -> List[str]:
    """Detect no-op changes where original == suggested.
    
    Returns:
        List of no-op descriptions
    """
    noops = []
    
    for suggestion in suggestions:
        for change in suggestion.changes:
            if change.original_code == change.suggested_code:
                noops.append(
                    f"No-op in {suggestion.file_path}: original equals suggested"
                )
            elif change.original_code.strip() == change.suggested_code.strip():
                noops.append(
                    f"Whitespace-only change in {suggestion.file_path}"
                )
    
    return noops


def add_validation_to_report(
    report: SuggestionReport,
    cycle_spec: CycleSpec,
) -> SuggestionReport:
    """Add validation results to a suggestion report.
    
    Args:
        report: The report to validate
        cycle_spec: The cycle specification
        
    Returns:
        Report with validation added
    """
    validation = validate_suggestions(report, cycle_spec)
    report.validation = validation
    report.cycle_will_be_broken = validation.cycle_logically_broken
    
    # Add validation notes to individual suggestions
    for suggestion in report.suggestions:
        if not suggestion.validation_notes:
            suggestion.validation_notes = []
        
        if suggestion.is_new_file:
            suggestion.validation_notes.append("✓ New file will be created")
        else:
            if any(c.line_start == 0 for c in suggestion.changes):
                suggestion.validation_notes.append("⚠️ Exact location not confirmed")
            else:
                suggestion.validation_notes.append("✓ Changes located in source")
    
    return report
