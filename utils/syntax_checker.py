"""Shared syntax validation utilities.

This module provides common syntax checking functions used by both
the Refactor and Validator agents to ensure consistent validation.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re

from utils.logging import get_logger

logger = get_logger("syntax_checker")


class SyntaxIssueType(Enum):
    """Types of syntax issues."""
    UNBALANCED_BRACES = "unbalanced_braces"
    UNBALANCED_PARENS = "unbalanced_parens"
    UNBALANCED_BRACKETS = "unbalanced_brackets"
    UNBALANCED_ANGLES = "unbalanced_angles"
    UNBALANCED_QUOTES = "unbalanced_quotes"
    TRUNCATED_OUTPUT = "truncated_output"
    INCOMPLETE_STATEMENT = "incomplete_statement"
    DUPLICATE_LINES = "duplicate_lines"


@dataclass
class SyntaxIssue:
    """A syntax issue found in code."""
    issue_type: SyntaxIssueType
    path: str
    message: str
    severity: str  # critical, major, minor
    line: Optional[int] = None  # Line number where issue was detected
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass 
class ValidationResult:
    """Result of validating a code block."""
    is_valid: bool
    issues: List[SyntaxIssue]
    confidence: float  # 0.0-1.0, how confident we are the code is valid
    
    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)
    
    @property
    def has_major(self) -> bool:
        return any(i.severity == "major" for i in self.issues)


def check_bracket_balance(content: str, path: str = "") -> List[SyntaxIssue]:
    """Check if brackets/braces are balanced in content.
    
    Args:
        content: The code content to check
        path: File path for error messages
        
    Returns:
        List of SyntaxIssue for any imbalances found
    """
    issues = []
    
    # Determine language from extension for angle bracket handling
    ext = path.split('.')[-1].lower() if '.' in path else ''
    check_angles = ext in ('cs', 'java', 'ts', 'tsx', 'cpp', 'hpp', 'h')
    
    bracket_pairs = [
        ('{', '}', 'braces', SyntaxIssueType.UNBALANCED_BRACES),
        ('(', ')', 'parentheses', SyntaxIssueType.UNBALANCED_PARENS),
        ('[', ']', 'brackets', SyntaxIssueType.UNBALANCED_BRACKETS),
    ]
    
    if check_angles:
        bracket_pairs.append(('<', '>', 'angle brackets', SyntaxIssueType.UNBALANCED_ANGLES))
    
    for open_char, close_char, name, issue_type in bracket_pairs:
        open_count = content.count(open_char)
        close_count = content.count(close_char)
        
        if open_count != close_count:
            diff = open_count - close_count
            severity = "critical" if abs(diff) > 1 else "major"
            
            issues.append(SyntaxIssue(
                issue_type=issue_type,
                path=path,
                message=f"Unbalanced {name}: {open_count} open, {close_count} close",
                severity=severity,
                details={
                    "open_count": open_count,
                    "close_count": close_count,
                    "difference": diff,
                }
            ))
    
    return issues


def check_truncation(content: str, path: str = "") -> Optional[SyntaxIssue]:
    """Check if content appears to be truncated.
    
    Args:
        content: The code content to check
        path: File path for error messages
        
    Returns:
        SyntaxIssue if truncation detected, None otherwise
    """
    if not content or len(content) < 10:
        return None
    
    text = content.rstrip()
    
    # Check for significant bracket imbalance (strong truncation indicator)
    open_braces = text.count('{')
    close_braces = text.count('}')
    
    if open_braces - close_braces > 2:
        return SyntaxIssue(
            issue_type=SyntaxIssueType.TRUNCATED_OUTPUT,
            path=path,
            message=f"Content appears truncated: {open_braces} open braces, {close_braces} close",
            severity="critical",
            details={"indicator": "brace_imbalance", "open": open_braces, "close": close_braces}
        )
    
    # Check for truncation patterns at end
    truncation_endings = [
        ('...', 'ellipsis'),
        ('// ...', 'comment_ellipsis'),
        ('/* ...', 'block_comment_ellipsis'),
        ('// more', 'more_comment'),
        ('// etc', 'etc_comment'),
        ('// continued', 'continued_comment'),
        ('// rest of', 'rest_of_comment'),
    ]
    
    for ending, indicator in truncation_endings:
        if text.endswith(ending):
            return SyntaxIssue(
                issue_type=SyntaxIssueType.TRUNCATED_OUTPUT,
                path=path,
                message=f"Content appears truncated: ends with '{ending}'",
                severity="critical",
                details={"indicator": indicator, "ending": ending}
            )
    
    # Check for incomplete statement endings
    incomplete_endings = [',', '(', '{', '[', ':']
    for ending in incomplete_endings:
        if text.endswith(ending) and not text.endswith('::'):  # Allow C++ scope
            # Check if this is likely incomplete vs intentional
            lines = text.split('\n')
            last_line = lines[-1].strip() if lines else ""
            
            # Trailing comma at end of file is suspicious
            if ending == ',' and not any(c in last_line for c in ['{', '[', '(']):
                return SyntaxIssue(
                    issue_type=SyntaxIssueType.INCOMPLETE_STATEMENT,
                    path=path,
                    message=f"Content may be incomplete: ends with '{ending}'",
                    severity="major",
                    details={"indicator": "incomplete_ending", "ending": ending}
                )
    
    # Check for unbalanced quotes (simple heuristic)
    double_quotes = text.count('"') - text.count('\\"')
    if double_quotes % 2 != 0:
        return SyntaxIssue(
            issue_type=SyntaxIssueType.UNBALANCED_QUOTES,
            path=path,
            message="Unbalanced double quotes detected",
            severity="major",
            details={"quote_count": double_quotes}
        )
    
    return None


def check_introduced_issues(
    original: str, 
    patched: str, 
    path: str = ""
) -> List[SyntaxIssue]:
    """Check if patching introduced new syntax issues.
    
    Compares original and patched content to identify issues
    that were introduced by the patch (not pre-existing).
    
    Args:
        original: Original file content
        patched: Patched file content  
        path: File path for error messages
        
    Returns:
        List of SyntaxIssue for issues introduced by the patch
    """
    issues = []
    
    if not patched:
        issues.append(SyntaxIssue(
            issue_type=SyntaxIssueType.TRUNCATED_OUTPUT,
            path=path,
            message="Patched content is empty",
            severity="critical",
            details={"original_length": len(original)}
        ))
        return issues
    
    # Check bracket balance changes
    ext = path.split('.')[-1].lower() if '.' in path else ''
    check_angles = ext in ('cs', 'java', 'ts', 'tsx', 'cpp', 'hpp', 'h')
    
    bracket_pairs = [
        ('{', '}', 'braces', SyntaxIssueType.UNBALANCED_BRACES),
        ('(', ')', 'parentheses', SyntaxIssueType.UNBALANCED_PARENS),
        ('[', ']', 'brackets', SyntaxIssueType.UNBALANCED_BRACKETS),
    ]
    
    if check_angles:
        bracket_pairs.append(('<', '>', 'angle brackets', SyntaxIssueType.UNBALANCED_ANGLES))
    
    for open_char, close_char, name, issue_type in bracket_pairs:
        orig_open = original.count(open_char)
        orig_close = original.count(close_char)
        patch_open = patched.count(open_char)
        patch_close = patched.count(close_char)
        
        orig_balanced = orig_open == orig_close
        patch_balanced = patch_open == patch_close
        
        if orig_balanced and not patch_balanced:
            # We introduced an imbalance!
            issues.append(SyntaxIssue(
                issue_type=issue_type,
                path=path,
                message=f"Patch introduced unbalanced {name}: was {orig_open}/{orig_close}, now {patch_open}/{patch_close}",
                severity="critical",
                details={
                    "original_open": orig_open,
                    "original_close": orig_close,
                    "patched_open": patch_open,
                    "patched_close": patch_close,
                    "introduced": True,
                }
            ))
        elif not orig_balanced and not patch_balanced:
            # Check if we made it worse
            orig_diff = abs(orig_open - orig_close)
            patch_diff = abs(patch_open - patch_close)
            
            if patch_diff > orig_diff:
                issues.append(SyntaxIssue(
                    issue_type=issue_type,
                    path=path,
                    message=f"Patch worsened {name} imbalance: was {orig_open}/{orig_close}, now {patch_open}/{patch_close}",
                    severity="major",
                    details={
                        "original_open": orig_open,
                        "original_close": orig_close,
                        "patched_open": patch_open,
                        "patched_close": patch_close,
                        "worsened": True,
                    }
                ))
    
    # Check for duplicate lines that weren't in original
    patched_lines = patched.split('\n')
    for i in range(len(patched_lines) - 1):
        line = patched_lines[i].strip()
        next_line = patched_lines[i + 1].strip()
        
        if line and line == next_line:
            # Check if this duplication existed in original
            dup_pattern = line + '\n' + line
            if dup_pattern not in original and line not in ('', '{', '}', 'break;', 'continue;', 'return;'):
                issues.append(SyntaxIssue(
                    issue_type=SyntaxIssueType.DUPLICATE_LINES,
                    path=path,
                    message=f"Patch introduced duplicate line at {i+1}: {line[:50]}...",
                    severity="minor",
                    details={"line_number": i + 1, "content": line[:100]}
                ))
                break  # Only report first
    
    return issues


def validate_code_block(
    content: str, 
    path: str = "",
    original: str = None
) -> ValidationResult:
    """Comprehensive validation of a code block.
    
    Args:
        content: The code content to validate
        path: File path for context
        original: Original content to compare against (optional)
        
    Returns:
        ValidationResult with all issues found
    """
    issues = []
    
    # Check for truncation first (most critical)
    truncation = check_truncation(content, path)
    if truncation:
        issues.append(truncation)
    
    # Check bracket balance
    balance_issues = check_bracket_balance(content, path)
    issues.extend(balance_issues)
    
    # If we have original, check for introduced issues
    if original is not None:
        introduced = check_introduced_issues(original, content, path)
        issues.extend(introduced)
    
    # Calculate confidence
    if any(i.severity == "critical" for i in issues):
        confidence = 0.0
    elif any(i.severity == "major" for i in issues):
        confidence = 0.3
    elif any(i.severity == "minor" for i in issues):
        confidence = 0.7
    else:
        confidence = 1.0
    
    is_valid = not any(i.severity in ("critical", "major") for i in issues)
    
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        confidence=confidence
    )


def get_common_indent(lines: List[str]) -> str:
    """Get the common leading whitespace from a list of lines.
    
    Args:
        lines: List of code lines
        
    Returns:
        Common indentation string
    """
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    
    indents = []
    for line in non_empty:
        match = re.match(r'^(\s*)', line)
        if match:
            indents.append(match.group(1))
    
    if not indents:
        return ""
    
    return min(indents, key=len)


def normalize_line_endings(content: str) -> str:
    """Normalize line endings to Unix style (LF).
    
    Args:
        content: Content with potentially mixed line endings
        
    Returns:
        Content with consistent LF line endings
    """
    return content.replace('\r\n', '\n').replace('\r', '\n')
