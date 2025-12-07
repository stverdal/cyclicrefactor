#!/usr/bin/env python3
"""Unit tests for utils.patch_parser module."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.patch_parser import (
    parse_json_patches,
    parse_marker_patches,
    parse_search_replace_json,
    clean_code_content,
    extract_patches_from_data,
    extract_json_string_value,
    infer_patches,
)


class TestParseJsonPatches:
    """Test the parse_json_patches function."""

    def test_clean_json(self):
        """Test parsing clean JSON without markdown wrapper."""
        json_text = '''
        {
            "patches": [
                {"path": "file1.cs", "patched": "content1"},
                {"path": "file2.cs", "patched": "content2"}
            ]
        }
        '''
        result = parse_json_patches(json_text)
        assert len(result) == 2
        assert result[0]["path"] == "file1.cs"
        assert result[0]["patched"] == "content1"
        assert result[1]["path"] == "file2.cs"

    def test_json_with_markdown_wrapper(self):
        """Test parsing JSON wrapped in markdown code block."""
        json_text = '''```json
{
    "patches": [
        {"path": "test.cs", "patched": "test content"}
    ]
}
```'''
        result = parse_json_patches(json_text)
        assert len(result) == 1
        assert result[0]["path"] == "test.cs"

    def test_json_with_embedded_code_blocks(self):
        """Test parsing JSON where patched content has code block markers."""
        json_text = '''
{
    "patches": [
        {
            "path": "example.cs",
            "patched": "```csharp\\npublic class Test { }\\n```"
        }
    ]
}
'''
        result = parse_json_patches(json_text)
        assert len(result) == 1
        assert "public class Test" in result[0]["patched"]
        assert "```" not in result[0]["patched"]

    def test_empty_patches_array(self):
        """Test JSON with empty patches array."""
        json_text = '{"patches": []}'
        result = parse_json_patches(json_text)
        assert len(result) == 0

    def test_missing_path_field(self):
        """Test patch missing path field is skipped."""
        json_text = '{"patches": [{"patched": "content"}]}'
        result = parse_json_patches(json_text)
        assert len(result) == 0

    def test_missing_patched_field(self):
        """Test patch missing patched field is skipped."""
        json_text = '{"patches": [{"path": "test.cs"}]}'
        result = parse_json_patches(json_text)
        assert len(result) == 0

    def test_invalid_json(self):
        """Test handling of completely invalid JSON."""
        result = parse_json_patches("not json at all")
        assert len(result) == 0


class TestParseMarkerPatches:
    """Test the parse_marker_patches function."""

    def test_single_file_marker(self):
        """Test parsing single file with marker."""
        text = """--- FILE: src/test.cs ---
public class Test {
    public void Method() { }
}"""
        result = parse_marker_patches(text)
        assert len(result) == 1
        assert result[0]["path"] == "src/test.cs"
        assert "public class Test" in result[0]["patched"]

    def test_multiple_file_markers(self):
        """Test parsing multiple files with markers."""
        text = """--- FILE: file1.cs ---
content1
--- FILE: file2.cs ---
content2"""
        result = parse_marker_patches(text)
        assert len(result) == 2
        assert result[0]["path"] == "file1.cs"
        assert result[1]["path"] == "file2.cs"

    def test_search_replace_in_marker(self):
        """Test marker format with embedded SEARCH/REPLACE blocks."""
        text = """--- FILE: test.cs ---
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE"""
        result = parse_marker_patches(text)
        assert len(result) == 1
        assert result[0]["search_replace_blocks"] is not None
        assert result[0]["patched"] is None

    def test_no_markers(self):
        """Test text without markers returns empty list."""
        result = parse_marker_patches("just some text without markers")
        assert len(result) == 0


class TestParseSearchReplaceJson:
    """Test the parse_search_replace_json function."""

    def test_valid_search_replace_json(self):
        """Test parsing valid search/replace JSON format."""
        text = '''
{
    "changes": [
        {
            "path": "test.cs",
            "search_replace": [
                {"search": "old1", "replace": "new1"},
                {"search": "old2", "replace": "new2"}
            ]
        }
    ]
}'''
        result = parse_search_replace_json(text)
        assert len(result) == 1
        assert result[0]["path"] == "test.cs"
        assert len(result[0]["search_replace"]) == 2

    def test_empty_changes(self):
        """Test JSON with empty changes array."""
        text = '{"changes": []}'
        result = parse_search_replace_json(text)
        assert len(result) == 0

    def test_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        text = '''```json
{"changes": [{"path": "a.cs", "search_replace": [{"search": "x", "replace": "y"}]}]}
```'''
        result = parse_search_replace_json(text)
        assert len(result) == 1


class TestCleanCodeContent:
    """Test the clean_code_content function."""

    def test_remove_csharp_block(self):
        """Test removing csharp code block markers."""
        content = "```csharp\npublic class Test { }\n```"
        result = clean_code_content(content)
        assert result == "public class Test { }"
        assert "```" not in result

    def test_remove_generic_block(self):
        """Test removing generic code block markers."""
        content = "```\nsome code\n```"
        result = clean_code_content(content)
        assert result == "some code"

    def test_no_markers(self):
        """Test content without markers is unchanged."""
        content = "plain content"
        result = clean_code_content(content)
        assert result == "plain content"

    def test_only_leading_marker(self):
        """Test content with only leading marker."""
        content = "```python\ncode here"
        result = clean_code_content(content)
        assert result == "code here"


class TestExtractJsonStringValue:
    """Test the extract_json_string_value function."""

    def test_simple_string(self):
        """Test extracting simple string value."""
        result = extract_json_string_value('hello world"')
        assert result == "hello world"

    def test_escaped_quotes(self):
        """Test handling escaped quotes."""
        result = extract_json_string_value('say \\"hello\\"",')
        assert result == 'say "hello"'

    def test_escaped_newlines(self):
        """Test handling escaped newlines."""
        result = extract_json_string_value('line1\\nline2"')
        assert result == "line1\nline2"

    def test_escaped_tabs(self):
        """Test handling escaped tabs."""
        result = extract_json_string_value('col1\\tcol2"')
        assert result == "col1\tcol2"


class TestExtractPatchesFromData:
    """Test the extract_patches_from_data function."""

    def test_valid_patches(self):
        """Test extracting valid patches from data dict."""
        data = {
            "patches": [
                {"path": "a.cs", "patched": "content a"},
                {"path": "b.cs", "patched": "content b"}
            ]
        }
        result = extract_patches_from_data(data)
        assert len(result) == 2

    def test_no_patches_key(self):
        """Test data without patches key."""
        data = {"other": "data"}
        result = extract_patches_from_data(data)
        assert len(result) == 0

    def test_patches_with_code_blocks(self):
        """Test patched content with code block markers gets cleaned."""
        data = {
            "patches": [
                {"path": "test.cs", "patched": "```csharp\ncode\n```"}
            ]
        }
        result = extract_patches_from_data(data)
        assert len(result) == 1
        assert "```" not in result[0]["patched"]


class TestInferPatches:
    """Test the infer_patches function."""

    def test_infer_json_format(self):
        """Test inferring patches from JSON format."""
        llm_response = '''{"patches": [{"path": "test.cs", "patched": "content"}]}'''
        cycle_files = [{"path": "test.cs", "content": "original"}]
        result = infer_patches(llm_response, cycle_files)
        assert len(result) == 1

    def test_infer_marker_format(self):
        """Test inferring patches from marker format."""
        llm_response = """--- FILE: test.cs ---
new content here"""
        cycle_files = [{"path": "test.cs", "content": "original"}]
        result = infer_patches(llm_response, cycle_files)
        assert len(result) == 1

    def test_infer_search_replace_format(self):
        """Test inferring patches from search/replace JSON format."""
        llm_response = '''{"changes": [{"path": "test.cs", "search_replace": [{"search": "old", "replace": "new"}]}]}'''
        cycle_files = [{"path": "test.cs", "content": "old code"}]
        result = infer_patches(llm_response, cycle_files)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
