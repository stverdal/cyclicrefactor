"""Syntax auto-repair utilities for fixing common LLM-generated code issues.

This module provides automated repair functions for common syntax errors
that LLMs produce when generating code patches:
- Unbalanced braces/brackets
- Trailing incomplete statements
- Missing closing elements at end of file
- Truncated code blocks

Auto-repair is conservative - it only attempts repairs when it has high
confidence the fix is correct. When uncertain, it returns the original
content unchanged.
"""

from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import re

from utils.logging import get_logger
from utils.syntax_checker import (
    check_bracket_balance, SyntaxIssue, SyntaxIssueType,
    normalize_line_endings
)

logger = get_logger("syntax_repair")


@dataclass
class RepairResult:
    """Result of an auto-repair attempt."""
    content: str
    was_repaired: bool
    repairs_made: List[str]  # Description of each repair
    confidence: float  # How confident we are the repair is correct (0.0-1.0)
    
    def __str__(self) -> str:
        if self.was_repaired:
            return f"Repaired ({len(self.repairs_made)} fixes, confidence={self.confidence:.0%})"
        return "No repairs needed"


def count_brackets(content: str, open_char: str, close_char: str) -> Tuple[int, int]:
    """Count open and close brackets, ignoring those in strings/comments.
    
    This is a simplified version that handles common cases.
    For production use, consider using a proper parser.
    """
    open_count = 0
    close_count = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape_next = False
    i = 0
    
    while i < len(content):
        char = content[i]
        next_char = content[i + 1] if i + 1 < len(content) else ''
        
        # Handle escape sequences
        if escape_next:
            escape_next = False
            i += 1
            continue
        
        if char == '\\' and (in_string or in_char):
            escape_next = True
            i += 1
            continue
        
        # Handle strings
        if char == '"' and not in_char and not in_line_comment and not in_block_comment:
            in_string = not in_string
            i += 1
            continue
        
        # Handle char literals
        if char == "'" and not in_string and not in_line_comment and not in_block_comment:
            in_char = not in_char
            i += 1
            continue
        
        # Handle line comments (//)
        if char == '/' and next_char == '/' and not in_string and not in_char and not in_block_comment:
            in_line_comment = True
            i += 2
            continue
        
        # Handle block comments (/* */)
        if char == '/' and next_char == '*' and not in_string and not in_char and not in_line_comment:
            in_block_comment = True
            i += 2
            continue
        
        if char == '*' and next_char == '/' and in_block_comment:
            in_block_comment = False
            i += 2
            continue
        
        # Handle newlines (end line comments)
        if char == '\n':
            in_line_comment = False
        
        # Count brackets (only when not in string/comment)
        if not in_string and not in_char and not in_line_comment and not in_block_comment:
            if char == open_char:
                open_count += 1
            elif char == close_char:
                close_count += 1
        
        i += 1
    
    return open_count, close_count


def repair_unbalanced_braces(content: str, path: str = "") -> RepairResult:
    """Attempt to repair unbalanced curly braces.
    
    Common LLM issues this fixes:
    1. Missing closing braces at end of file (truncated output)
    2. Extra opening brace from copy-paste errors
    3. Missing opening brace (less common, harder to fix)
    
    Args:
        content: The code content with potential brace issues
        path: File path for context (used to determine language)
        
    Returns:
        RepairResult with repaired content if successful
    """
    content = normalize_line_endings(content)
    repairs = []
    
    open_count, close_count = count_brackets(content, '{', '}')
    
    if open_count == close_count:
        return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)
    
    diff = open_count - close_count
    
    # Case 1: More opens than closes (most common LLM error - truncated output)
    if diff > 0:
        # Only repair if the imbalance is reasonable (1-3 missing braces)
        if diff > 3:
            logger.warning(f"Too many missing braces ({diff}) - not safe to auto-repair")
            return RepairResult(
                content=content, was_repaired=False, 
                repairs_made=[f"Detected {diff} missing closing braces, too many to safely repair"],
                confidence=0.0
            )
        
        # Check if the file ends in a reasonable way for adding braces
        lines = content.rstrip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        # Determine appropriate indentation for closing braces
        # Find the indentation level by looking at the last few lines
        base_indent = ""
        for line in reversed(lines[-10:]):
            if line.strip() and not line.strip().startswith('//'):
                # Get indentation of this line
                match = re.match(r'^(\s*)', line)
                if match:
                    base_indent = match.group(1)
                    break
        
        # Add the missing closing braces
        closing_braces = []
        current_indent = base_indent
        
        for i in range(diff):
            # Each closing brace should be at a decreasing indent level
            # Assume 4-space or 1-tab indentation
            if current_indent.startswith('\t'):
                indent_unit = '\t'
            else:
                indent_unit = '    '
            
            # Decrease indent for each brace
            if current_indent.startswith(indent_unit):
                current_indent = current_indent[len(indent_unit):]
            
            closing_braces.append(f"{current_indent}}}")
        
        # Add braces at the end
        repaired = content.rstrip() + '\n' + '\n'.join(closing_braces) + '\n'
        
        repairs.append(f"Added {diff} missing closing brace(s) at end of file")
        logger.info(f"Auto-repair: Added {diff} closing braces to {path}")
        
        # Confidence is higher for fewer missing braces
        confidence = 0.9 if diff == 1 else (0.75 if diff == 2 else 0.6)
        
        return RepairResult(
            content=repaired,
            was_repaired=True,
            repairs_made=repairs,
            confidence=confidence
        )
    
    # Case 2: More closes than opens (less common, harder to fix)
    elif diff < 0:
        extra_closes = abs(diff)
        
        if extra_closes > 2:
            logger.warning(f"Too many extra closing braces ({extra_closes}) - not safe to auto-repair")
            return RepairResult(
                content=content, was_repaired=False,
                repairs_made=[f"Detected {extra_closes} extra closing braces, too many to safely repair"],
                confidence=0.0
            )
        
        # Try to find and remove orphan closing braces
        # Look for }} at end of file that might be duplicates
        lines = content.split('\n')
        repaired_lines = lines.copy()
        removed = 0
        
        # Check last few lines for orphan braces
        for i in range(len(repaired_lines) - 1, max(0, len(repaired_lines) - 5), -1):
            line = repaired_lines[i]
            if line.strip() == '}' and removed < extra_closes:
                # Check if removing this brace would help
                test_content = '\n'.join(repaired_lines[:i] + repaired_lines[i+1:])
                test_open, test_close = count_brackets(test_content, '{', '}')
                if abs(test_open - test_close) < abs(diff) - removed:
                    repaired_lines.pop(i)
                    removed += 1
                    repairs.append(f"Removed orphan closing brace at line {i + 1}")
        
        if removed > 0:
            repaired = '\n'.join(repaired_lines)
            logger.info(f"Auto-repair: Removed {removed} extra closing braces from {path}")
            return RepairResult(
                content=repaired,
                was_repaired=True,
                repairs_made=repairs,
                confidence=0.7
            )
    
    return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=0.5)


def repair_unbalanced_parens(content: str, path: str = "") -> RepairResult:
    """Attempt to repair unbalanced parentheses.
    
    This is more conservative than brace repair since parentheses are
    used in many contexts (function calls, conditions, expressions).
    
    Only repairs obvious cases at end of file.
    """
    content = normalize_line_endings(content)
    
    open_count, close_count = count_brackets(content, '(', ')')
    
    if open_count == close_count:
        return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)
    
    diff = open_count - close_count
    
    # Only repair 1-2 missing closing parens at end of file
    if diff > 0 and diff <= 2:
        lines = content.rstrip().split('\n')
        last_line = lines[-1] if lines else ""
        
        # Check if the last line looks truncated (ends with comma, operator, etc.)
        truncation_indicators = [',', '(', '+', '-', '*', '/', '&&', '||', '?', ':']
        
        looks_truncated = any(last_line.rstrip().endswith(ind) for ind in truncation_indicators)
        
        if looks_truncated:
            # Add closing parens
            repaired = content.rstrip() + ')' * diff
            
            # Check if we need a semicolon too (C-family languages)
            ext = path.split('.')[-1].lower() if '.' in path else ''
            if ext in ('cs', 'java', 'ts', 'js', 'cpp', 'c', 'h'):
                if not repaired.rstrip().endswith(';') and not repaired.rstrip().endswith('{'):
                    repaired = repaired + ';'
            
            repaired = repaired + '\n'
            
            return RepairResult(
                content=repaired,
                was_repaired=True,
                repairs_made=[f"Added {diff} missing closing parenthesis at end of file"],
                confidence=0.6
            )
    
    return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=0.5)


def repair_trailing_comma(content: str, path: str = "") -> RepairResult:
    """Remove trailing comma at end of file (common truncation artifact)."""
    content = normalize_line_endings(content)
    
    # Check if file ends with a trailing comma (not inside a collection)
    lines = content.rstrip().split('\n')
    if not lines:
        return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)
    
    last_line = lines[-1].rstrip()
    
    # Only fix if the line ends with a lone comma (not part of an array/object literal)
    if last_line.endswith(','):
        # Check it's not inside an array or object
        open_braces, close_braces = count_brackets(content, '{', '}')
        open_brackets, close_brackets = count_brackets(content, '[', ']')
        
        # If braces/brackets are balanced, the trailing comma is likely an error
        if open_braces == close_braces and open_brackets == close_brackets:
            # Remove the trailing comma
            lines[-1] = last_line[:-1]
            repaired = '\n'.join(lines) + '\n'
            
            return RepairResult(
                content=repaired,
                was_repaired=True,
                repairs_made=["Removed trailing comma at end of file"],
                confidence=0.8
            )
    
    return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)


def repair_incomplete_statement(content: str, path: str = "") -> RepairResult:
    """Attempt to complete obviously truncated statements.
    
    Only handles very obvious cases to avoid breaking valid code.
    """
    content = normalize_line_endings(content)
    repairs = []
    
    ext = path.split('.')[-1].lower() if '.' in path else ''
    is_c_family = ext in ('cs', 'java', 'ts', 'js', 'cpp', 'c', 'h', 'tsx', 'jsx')
    
    lines = content.rstrip().split('\n')
    if not lines:
        return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)
    
    last_line = lines[-1].rstrip()
    
    # Check for incomplete statements in C-family languages
    if is_c_family:
        # Missing semicolon after a complete-looking statement
        if (last_line and 
            not last_line.endswith((';', '{', '}', ':', ',', '(', '//', '*/')) and
            not last_line.strip().startswith(('//', '/*', '*'))):
            
            # Check if this looks like a complete statement
            # (has a closing paren or ends with a value/identifier)
            if re.search(r'[\)\w\"\']$', last_line):
                lines[-1] = last_line + ';'
                repairs.append("Added missing semicolon at end of file")
    
    if repairs:
        repaired = '\n'.join(lines) + '\n'
        return RepairResult(
            content=repaired,
            was_repaired=True,
            repairs_made=repairs,
            confidence=0.7
        )
    
    return RepairResult(content=content, was_repaired=False, repairs_made=[], confidence=1.0)


def auto_repair_syntax(content: str, path: str = "", max_repairs: int = 5) -> RepairResult:
    """Apply all safe auto-repairs to content.
    
    Runs repair functions in order of reliability:
    1. Trailing comma removal (highest confidence)
    2. Unbalanced braces (common LLM error)
    3. Unbalanced parentheses (less common)
    4. Incomplete statements (lowest confidence)
    
    Args:
        content: The code content to repair
        path: File path for context
        max_repairs: Maximum number of repair passes to attempt
        
    Returns:
        RepairResult with all repairs applied
    """
    all_repairs = []
    min_confidence = 1.0
    current_content = content
    
    # Track iterations to prevent infinite loops
    for iteration in range(max_repairs):
        made_repair = False
        
        # 1. Trailing comma
        result = repair_trailing_comma(current_content, path)
        if result.was_repaired:
            current_content = result.content
            all_repairs.extend(result.repairs_made)
            min_confidence = min(min_confidence, result.confidence)
            made_repair = True
        
        # 2. Unbalanced braces
        result = repair_unbalanced_braces(current_content, path)
        if result.was_repaired:
            current_content = result.content
            all_repairs.extend(result.repairs_made)
            min_confidence = min(min_confidence, result.confidence)
            made_repair = True
        
        # 3. Unbalanced parentheses
        result = repair_unbalanced_parens(current_content, path)
        if result.was_repaired:
            current_content = result.content
            all_repairs.extend(result.repairs_made)
            min_confidence = min(min_confidence, result.confidence)
            made_repair = True
        
        # 4. Incomplete statements
        result = repair_incomplete_statement(current_content, path)
        if result.was_repaired:
            current_content = result.content
            all_repairs.extend(result.repairs_made)
            min_confidence = min(min_confidence, result.confidence)
            made_repair = True
        
        if not made_repair:
            break
    
    # Verify the repairs actually fixed the issues
    final_issues = check_bracket_balance(current_content, path)
    critical_remaining = [i for i in final_issues if i.severity == "critical"]
    
    if all_repairs:
        logger.info(f"Auto-repair for {path}: {len(all_repairs)} repairs, "
                   f"confidence={min_confidence:.0%}, remaining_issues={len(critical_remaining)}")
        for repair in all_repairs:
            logger.debug(f"  - {repair}")
    
    return RepairResult(
        content=current_content,
        was_repaired=len(all_repairs) > 0,
        repairs_made=all_repairs,
        confidence=min_confidence if all_repairs else 1.0
    )


def should_attempt_repair(issues: List[SyntaxIssue]) -> bool:
    """Determine if auto-repair should be attempted based on issues.
    
    Returns True only if:
    - There are syntax issues present
    - The issues are of types we can potentially repair
    - The number of issues is reasonable (not too many)
    """
    if not issues:
        return False
    
    repairable_types = {
        SyntaxIssueType.UNBALANCED_BRACES,
        SyntaxIssueType.UNBALANCED_PARENS,
        SyntaxIssueType.INCOMPLETE_STATEMENT,
    }
    
    repairable_issues = [i for i in issues if i.issue_type in repairable_types]
    
    # Only attempt if we have repairable issues and not too many
    if repairable_issues and len(repairable_issues) <= 3:
        return True
    
    return False
