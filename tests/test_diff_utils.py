#!/usr/bin/env python3
"""Unit tests for utils.diff_utils module."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.diff_utils import (
    make_unified_diff,
    looks_truncated,
    get_common_indent,
    validate_patched_content,
    check_for_truncation,
    compute_diff_stats,
)


class TestMakeUnifiedDiff:
    """Test the make_unified_diff function."""

    def test_simple_diff(self):
        """Test generating a simple diff."""
        original = "line 1\nline 2\nline 3"
        patched = "line 1\nmodified line\nline 3"
        
        diff = make_unified_diff(original, patched, "test.txt")
        
        assert "test.txt" in diff
        assert "-line 2" in diff
        assert "+modified line" in diff

    def test_no_changes(self):
        """Test diff with no changes returns empty string."""
        content = "unchanged content"
        diff = make_unified_diff(content, content, "test.txt")
        assert diff == ""

    def test_addition(self):
        """Test diff with added lines."""
        original = "line 1\nline 2"
        patched = "line 1\nnew line\nline 2"
        
        diff = make_unified_diff(original, patched, "test.txt")
        
        assert "+new line" in diff

    def test_deletion(self):
        """Test diff with deleted lines."""
        original = "line 1\ndelete me\nline 2"
        patched = "line 1\nline 2"
        
        diff = make_unified_diff(original, patched, "test.txt")
        
        assert "-delete me" in diff

    def test_context_lines(self):
        """Test that context lines are included."""
        original = "a\nb\nc\nd\ne"
        patched = "a\nb\nX\nd\ne"
        
        diff = make_unified_diff(original, patched, "test.txt", context_lines=1)
        
        # Should have context around the change
        assert "b" in diff
        assert "d" in diff


class TestLooksTruncated:
    """Test the looks_truncated function."""

    def test_not_truncated(self):
        """Test complete code is not flagged."""
        code = """
public class Complete {
    public void Method() {
        Console.WriteLine("Hello");
    }
}
"""
        assert looks_truncated(code) is False

    def test_truncated_ellipsis(self):
        """Test ellipsis pattern is detected."""
        code = """
public class Incomplete {
    public void Method() {
        // ... rest of method
"""
        assert looks_truncated(code) is True

    def test_truncated_continuation(self):
        """Test continuation comment is detected."""
        code = """
public void Method() {
    DoSomething();
    // ... (remaining implementation)
"""
        assert looks_truncated(code) is True

    def test_truncated_unbalanced_braces(self):
        """Test unbalanced braces are detected."""
        code = """
public class Unbalanced {
    public void Method() {
        if (condition) {
            DoThis();
"""
        assert looks_truncated(code) is True

    def test_short_text_not_flagged(self):
        """Test very short text is not flagged as truncated."""
        code = "short"
        assert looks_truncated(code) is False


class TestGetCommonIndent:
    """Test the get_common_indent function."""

    def test_consistent_indent(self):
        """Test lines with consistent indentation."""
        lines = ["    line 1", "    line 2", "    line 3"]
        result = get_common_indent(lines)
        assert result == "    "

    def test_mixed_indent(self):
        """Test lines with mixed indentation returns minimum."""
        lines = ["  line 1", "    line 2", "      line 3"]
        result = get_common_indent(lines)
        assert result == "  "

    def test_no_indent(self):
        """Test lines without indentation."""
        lines = ["line 1", "line 2"]
        result = get_common_indent(lines)
        assert result == ""

    def test_empty_lines_ignored(self):
        """Test that empty lines are ignored."""
        lines = ["    line 1", "", "    line 2"]
        result = get_common_indent(lines)
        assert result == "    "

    def test_empty_input(self):
        """Test empty input returns empty string."""
        result = get_common_indent([])
        assert result == ""


class TestValidatePatchedContent:
    """Test the validate_patched_content function."""

    def test_valid_patch(self):
        """Test valid patch returns no warnings."""
        original = "class Test { void Method() { } }"
        patched = "class Test { void NewMethod() { } }"
        
        warnings = validate_patched_content(original, patched, "test.cs")
        
        assert len(warnings) == 0

    def test_empty_patched_content(self):
        """Test empty patched content is flagged."""
        original = "some content"
        patched = ""
        
        warnings = validate_patched_content(original, patched, "test.cs")
        
        assert len(warnings) > 0
        assert "empty" in warnings[0].lower()

    def test_introduced_brace_imbalance(self):
        """Test introduced brace imbalance is detected."""
        original = "class Test { void Method() { } }"
        patched = "class Test { void Method() { }"  # Missing closing brace
        
        warnings = validate_patched_content(original, patched, "test.cs")
        
        assert len(warnings) > 0
        assert "brace" in warnings[0].lower() or "unbalanced" in warnings[0].lower()

    def test_introduced_paren_imbalance(self):
        """Test introduced parenthesis imbalance is detected."""
        original = "func(a, b)"
        patched = "func(a, b"  # Missing closing paren
        
        warnings = validate_patched_content(original, patched, "test.cs")
        
        assert len(warnings) > 0


class TestCheckForTruncation:
    """Test the check_for_truncation function."""

    def test_no_truncation(self):
        """Test complete content returns False."""
        content = """
public class Complete {
    public void Method() {
        return;
    }
}
"""
        result = check_for_truncation(content, "test.cs")
        assert result is False

    def test_ellipsis_truncation(self):
        """Test ellipsis truncation is detected."""
        content = """
public void Method() {
    // ... more code here
"""
        result = check_for_truncation(content, "test.cs")
        assert result is True

    def test_brace_truncation(self):
        """Test unbalanced braces truncation is detected."""
        content = """
public class Broken {
    public void Method() {
        if (true) {
            if (true) {
                if (true) {
                    if (true) {
"""
        result = check_for_truncation(content, "test.cs")
        assert result is True


class TestComputeDiffStats:
    """Test the compute_diff_stats function."""

    def test_simple_stats(self):
        """Test computing stats for a simple change."""
        original = "line 1\nline 2\nline 3"
        patched = "line 1\nnew line\nline 3"
        
        stats = compute_diff_stats(original, patched)
        
        assert stats["lines_added"] >= 1
        assert stats["lines_removed"] >= 1

    def test_no_change_stats(self):
        """Test stats for no changes."""
        content = "unchanged"
        stats = compute_diff_stats(content, content)
        
        assert stats["lines_added"] == 0
        assert stats["lines_removed"] == 0

    def test_pure_addition(self):
        """Test stats for pure addition."""
        original = "line 1\nline 2"
        patched = "line 1\nline 2\nline 3"
        
        stats = compute_diff_stats(original, patched)
        
        assert stats["lines_added"] >= 1
        assert stats["lines_removed"] == 0

    def test_pure_deletion(self):
        """Test stats for pure deletion."""
        original = "line 1\nline 2\nline 3"
        patched = "line 1\nline 3"
        
        stats = compute_diff_stats(original, patched)
        
        assert stats["lines_removed"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
