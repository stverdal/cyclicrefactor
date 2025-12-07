"""Diff and text comparison utilities.

This module provides functions for:
- Generating unified diffs
- Detecting truncated content
- Computing common indentation
- Validating patched content

Extracted from refactor_agent.py for better separation of concerns.
"""

from typing import List, Tuple
import difflib
import re

from utils.logging import get_logger
from utils.syntax_checker import check_truncation

logger = get_logger("diff_utils")


def make_unified_diff(original: str, patched: str, path: str) -> str:
    """Generate a unified diff between original and patched content.
    
    Args:
        original: Original file content
        patched: Patched file content
        path: File path for diff headers
        
    Returns:
        Unified diff string
    """
    orig_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, patched_lines, 
        fromfile=f"a/{path}", tofile=f"b/{path}"
    )
    return "".join(diff)


def looks_truncated(text: str) -> bool:
    """Check if text appears to be truncated (incomplete code).
    
    Delegates to shared syntax_checker.check_truncation for consistency.
    
    Args:
        text: Content to check
        
    Returns:
        True if the text shows signs of truncation
    """
    if not text or len(text) < 10:
        return False
    
    truncation_issue = check_truncation(text, "")
    return truncation_issue is not None


def get_common_indent(lines: List[str]) -> str:
    """Get the common leading whitespace from a list of lines.
    
    Args:
        lines: List of code lines
        
    Returns:
        Common leading whitespace string
    """
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    
    # Get leading whitespace from each non-empty line
    indents = []
    for line in non_empty:
        match = re.match(r'^(\s*)', line)
        if match:
            indents.append(match.group(1))
    
    if not indents:
        return ""
    
    # Return the shortest common prefix
    min_indent = min(indents, key=len)
    return min_indent


def validate_patched_content(original: str, patched: str, path: str) -> List[str]:
    """Validate that patched content doesn't have obvious syntax issues.
    
    Checks for:
    - Empty content
    - Unbalanced brackets
    - Duplicate lines (possible copy-paste errors)
    
    Args:
        original: Original file content
        patched: Patched file content
        path: File path for error messages
    
    Returns:
        List of validation warnings
    """
    warnings = []
    
    if not patched:
        warnings.append(f"{path}: Patched content is empty")
        return warnings
    
    # Check bracket balance
    brackets = [
        ('{', '}', 'braces'),
        ('(', ')', 'parentheses'),
        ('[', ']', 'brackets'),
    ]
    
    ext = path.split('.')[-1].lower() if '.' in path else ''
    if ext in ('cs', 'java', 'ts', 'tsx'):
        brackets.append(('<', '>', 'angle brackets'))
    
    for open_char, close_char, name in brackets:
        open_count = patched.count(open_char)
        close_count = patched.count(close_char)
        
        if open_count != close_count:
            # Check if original also had imbalance
            orig_open = original.count(open_char)
            orig_close = original.count(close_char)
            
            if orig_open == orig_close:
                # We introduced the imbalance!
                warning = f"{path}: Introduced unbalanced {name} ({open_count} open, {close_count} close)"
                logger.error(warning)
                warnings.append(warning)
            elif abs(open_count - close_count) > abs(orig_open - orig_close):
                # We made existing imbalance worse
                warning = f"{path}: Worsened {name} imbalance ({open_count} open, {close_count} close, was {orig_open}/{orig_close})"
                logger.warning(warning)
                warnings.append(warning)
    
    # Check for common syntax issues
    lines = patched.split('\n')
    
    # Look for duplicate lines that suggest copy-paste errors
    for i in range(len(lines) - 1):
        if lines[i].strip() and lines[i] == lines[i + 1]:
            # Check if original had this duplication
            if lines[i] + '\n' + lines[i] not in original:
                warning = f"{path}: Possible duplicate line at line {i + 1}: {lines[i][:50]}..."
                logger.warning(warning)
                warnings.append(warning)
                break  # Only report first
    
    return warnings


def check_for_truncation(content: str, path: str) -> bool:
    """Check if content appears to be truncated (unbalanced brackets).
    
    This is a legacy interface for compatibility. Use looks_truncated() instead.
    
    Args:
        content: Content to check
        path: File path (for logging)
    
    Returns:
        True if truncation is detected
    """
    # Common truncation indicators
    truncation_markers = [
        "...",
        "// ...",
        "/* ... */",
        "# ...",
        "// truncated",
        "// remaining code",
        "// rest of",
        "// etc.",
        "// ... (remaining",
        "// more code",
        "... (and so on)",
        "... (continued)",
    ]
    
    content_lower = content.lower()
    lines = content.split('\n')
    
    # Check last few lines for truncation markers
    for line in lines[-5:]:
        line_stripped = line.strip().lower()
        for marker in truncation_markers:
            if marker in line_stripped:
                logger.warning(f"{path}: Truncation marker detected: '{line.strip()}'")
                return True
    
    # Check bracket balance as a sign of truncation
    brackets = [
        ('{', '}'),
        ('(', ')'),
        ('[', ']'),
    ]
    
    for open_char, close_char in brackets:
        open_count = content.count(open_char)
        close_count = content.count(close_char)
        
        # Significant imbalance suggests truncation
        if open_count - close_count > 3:
            logger.warning(f"{path}: Significant bracket imbalance ({open_count} '{open_char}' vs {close_count} '{close_char}'), likely truncated")
            return True
    
    return False


def compute_diff_stats(original: str, patched: str) -> dict:
    """Compute statistics about the differences between original and patched.
    
    Args:
        original: Original file content
        patched: Patched file content
        
    Returns:
        Dict with stats like lines_added, lines_removed, lines_changed
    """
    orig_lines = original.splitlines()
    patch_lines = patched.splitlines()
    
    matcher = difflib.SequenceMatcher(None, orig_lines, patch_lines)
    
    lines_added = 0
    lines_removed = 0
    lines_unchanged = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            lines_removed += i2 - i1
            lines_added += j2 - j1
        elif tag == 'delete':
            lines_removed += i2 - i1
        elif tag == 'insert':
            lines_added += j2 - j1
        elif tag == 'equal':
            lines_unchanged += i2 - i1
    
    return {
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "lines_unchanged": lines_unchanged,
        "total_original": len(orig_lines),
        "total_patched": len(patch_lines),
        "similarity_ratio": matcher.ratio(),
    }
