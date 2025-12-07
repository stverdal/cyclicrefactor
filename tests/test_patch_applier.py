#!/usr/bin/env python3
"""Unit tests for utils.patch_applier module."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.patch_applier import (
    SearchReplaceResult,
    MatchResult,
    apply_search_replace_atomic,
    apply_search_replace_list_atomic,
    try_find_search_text,
    extract_line_hint,
)


class TestSearchReplaceResult:
    """Test the SearchReplaceResult dataclass."""

    def test_creation(self):
        """Test creating a SearchReplaceResult."""
        result = SearchReplaceResult(
            content="new content",
            applied_count=2,
            total_count=3,
            warnings=["warning1"],
            confidence=0.85,
            is_atomic=True
        )
        assert result.content == "new content"
        assert result.applied_count == 2
        assert result.total_count == 3
        assert len(result.warnings) == 1
        assert result.confidence == 0.85
        assert result.is_atomic is True

    def test_defaults(self):
        """Test default values."""
        result = SearchReplaceResult(content="x")
        assert result.applied_count == 0
        assert result.total_count == 0
        assert result.warnings == []
        assert result.confidence == 0.0
        assert result.is_atomic is True


class TestMatchResult:
    """Test the MatchResult dataclass."""

    def test_creation(self):
        """Test creating a MatchResult."""
        result = MatchResult(
            start=10,
            end=20,
            strategy="exact",
            confidence=1.0
        )
        assert result.start == 10
        assert result.end == 20
        assert result.strategy == "exact"
        assert result.confidence == 1.0


class TestExtractLineHint:
    """Test the extract_line_hint function."""

    def test_line_comment(self):
        """Test extracting line from comment like '// Line 42'."""
        result = extract_line_hint("// Line 42")
        assert result == 41  # 0-indexed

    def test_around_line(self):
        """Test extracting from 'around line' pattern."""
        result = extract_line_hint("// around line 100")
        assert result == 99

    def test_near_line(self):
        """Test extracting from 'near line' pattern."""
        result = extract_line_hint("# near line 50")
        assert result == 49

    def test_parenthetical(self):
        """Test extracting from parenthetical pattern."""
        result = extract_line_hint("some code (line 25)")
        assert result == 24

    def test_no_hint(self):
        """Test text without line hint returns None."""
        result = extract_line_hint("no line number here")
        assert result is None


class TestTryFindSearchText:
    """Test the try_find_search_text function."""

    def test_exact_match(self):
        """Test exact line-by-line match."""
        content_lines = ["line 1", "line 2", "line 3", "line 4"]
        content_stripped = [l.rstrip() for l in content_lines]
        search_text = "line 2\nline 3"
        
        result = try_find_search_text(search_text, content_lines, content_stripped)
        
        assert result.start == 1
        assert result.end == 3
        assert result.strategy == "exact"
        assert result.confidence == 1.0

    def test_exact_match_with_whitespace(self):
        """Test exact match ignoring trailing whitespace."""
        content_lines = ["line 1  ", "line 2  ", "line 3"]
        content_stripped = [l.rstrip() for l in content_lines]
        search_text = "line 1\nline 2"
        
        result = try_find_search_text(search_text, content_lines, content_stripped)
        
        assert result.start == 0
        assert result.end == 2
        assert result.confidence == 1.0

    def test_multiple_exact_matches_with_hint(self):
        """Test disambiguation using line hint for multiple matches."""
        content_lines = ["func()", "body", "func()", "body"]
        content_stripped = content_lines.copy()
        search_text = "func()\nbody"
        
        # With hint pointing to second occurrence
        result = try_find_search_text(search_text, content_lines, content_stripped, line_hint=2)
        
        assert result.start == 2
        assert result.strategy == "exact_with_hint"
        assert result.confidence == 0.95

    def test_no_match(self):
        """Test text that doesn't match returns empty result."""
        content_lines = ["line 1", "line 2"]
        content_stripped = content_lines.copy()
        search_text = "not found"
        
        result = try_find_search_text(search_text, content_lines, content_stripped)
        
        assert result.start is None
        assert result.end is None
        assert result.strategy == ""
        assert result.confidence == 0.0


class TestApplySearchReplaceAtomic:
    """Test the apply_search_replace_atomic function."""

    def test_single_block_success(self):
        """Test applying a single SEARCH/REPLACE block."""
        original = "line 1\nold text\nline 3"
        blocks = """<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE"""
        
        result = apply_search_replace_atomic(original, blocks)
        
        assert result.applied_count == 1
        assert result.total_count == 1
        assert "new text" in result.content
        assert "old text" not in result.content
        assert result.is_atomic is True

    def test_multiple_blocks_success(self):
        """Test applying multiple SEARCH/REPLACE blocks."""
        original = "AAA\nBBB\nCCC"
        blocks = """<<<<<<< SEARCH
AAA
=======
111
>>>>>>> REPLACE

<<<<<<< SEARCH
CCC
=======
333
>>>>>>> REPLACE"""
        
        result = apply_search_replace_atomic(original, blocks)
        
        assert result.applied_count == 2
        assert result.total_count == 2
        assert result.content == "111\nBBB\n333"

    def test_atomic_failure_missing_search(self):
        """Test atomic failure when search text not found."""
        original = "existing content"
        blocks = """<<<<<<< SEARCH
not in file
=======
replacement
>>>>>>> REPLACE"""
        
        result = apply_search_replace_atomic(original, blocks)
        
        assert result.applied_count == 0
        assert result.total_count == 1
        assert result.content == original  # Original returned on failure
        assert len(result.warnings) > 0

    def test_no_blocks_found(self):
        """Test when no valid blocks are parsed."""
        original = "content"
        blocks = "not a valid block format"
        
        result = apply_search_replace_atomic(original, blocks)
        
        assert result.applied_count == 0
        assert result.total_count == 0

    def test_confidence_threshold(self):
        """Test that low confidence matches are rejected by default."""
        original = "function start\nsome code\nfunction end"
        # This search text won't match exactly
        blocks = """<<<<<<< SEARCH
function START
some CODE
function END
=======
replacement
>>>>>>> REPLACE"""
        
        result = apply_search_replace_atomic(original, blocks, min_confidence=0.9)
        
        # Should fail because case doesn't match
        assert result.applied_count == 0

    def test_preserve_indentation(self):
        """Test that indentation is preserved during replacement."""
        original = "    indented line\n    another indented"
        blocks = """<<<<<<< SEARCH
    indented line
=======
changed line
>>>>>>> REPLACE"""
        
        result = apply_search_replace_atomic(original, blocks)
        
        # The replacement should preserve the original indent
        assert result.applied_count == 1


class TestApplySearchReplaceListAtomic:
    """Test the apply_search_replace_list_atomic function."""

    def test_single_operation(self):
        """Test applying a single search/replace operation."""
        original = "old value here"
        operations = [{"search": "old value", "replace": "new value"}]
        
        result = apply_search_replace_list_atomic(original, operations)
        
        assert result.applied_count == 1
        assert "new value" in result.content

    def test_multiple_operations(self):
        """Test applying multiple operations."""
        original = "AAA and BBB"
        operations = [
            {"search": "AAA", "replace": "111"},
            {"search": "BBB", "replace": "222"}
        ]
        
        result = apply_search_replace_list_atomic(original, operations)
        
        assert result.applied_count == 2
        assert "111" in result.content
        assert "222" in result.content

    def test_empty_list(self):
        """Test with empty operations list."""
        original = "content"
        operations = []
        
        result = apply_search_replace_list_atomic(original, operations)
        
        assert result.applied_count == 0
        assert result.content == original

    def test_line_hint_from_dict(self):
        """Test that line hint can be extracted from operation dict."""
        original = "pattern\nother\npattern\nmore"
        operations = [
            {"search": "pattern", "replace": "changed", "line": 3}  # Target second occurrence
        ]
        
        result = apply_search_replace_list_atomic(original, operations)
        
        # Should apply to second 'pattern' at line 3
        assert result.applied_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
