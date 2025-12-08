"""Line number-based patching utilities.

This module provides patching by line numbers instead of text search/replace.
Benefits:
- More reliable: exact location, no ambiguity
- Handles whitespace differences
- Better error messages
- Works even with partial LLM output
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("line_patch")


@dataclass
class LinePatch:
    """A single line-based patch operation."""
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    new_content: str
    operation: str = "replace"  # "replace", "insert", "delete"
    description: str = ""


@dataclass 
class LinePatchResult:
    """Result of applying line patches."""
    content: str
    success: bool
    applied_count: int
    total_count: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """Add line numbers to content for LLM context.
    
    Args:
        content: File content
        start_line: Starting line number (1-indexed)
        
    Returns:
        Content with line numbers prefixed: "  42 | code here"
    """
    lines = content.splitlines()
    numbered = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        # Right-align line numbers to 4 chars
        numbered.append(f"{line_num:4d} | {line}")
    return "\n".join(numbered)


def build_numbered_file_snippets(
    files: List[Dict[str, Any]],
    max_chars_per_file: int = 8000,
) -> Tuple[str, Dict[str, int]]:
    """Build file snippets with line numbers for LLM prompt.
    
    Args:
        files: List of file dicts with 'path' and 'content' keys
        max_chars_per_file: Maximum chars per file
        
    Returns:
        Tuple of (combined_snippets_string, file_line_counts)
    """
    snippets = []
    line_counts = {}
    
    for f in files:
        path = f.get('path', 'unknown')
        content = f.get('content', '') or ''
        
        if not content:
            line_counts[path] = 0
            continue
        
        lines = content.splitlines()
        line_counts[path] = len(lines)
        
        # Truncate if needed, but show where we truncated
        if len(content) > max_chars_per_file:
            # Try to include beginning and end
            char_budget = max_chars_per_file - 100  # Reserve for truncation note
            begin_budget = int(char_budget * 0.6)
            end_budget = int(char_budget * 0.4)
            
            begin_content = content[:begin_budget]
            # Find last complete line
            last_newline = begin_content.rfind('\n')
            if last_newline > 0:
                begin_content = begin_content[:last_newline]
            begin_lines = begin_content.splitlines()
            
            end_content = content[-end_budget:]
            # Find first complete line
            first_newline = end_content.find('\n')
            if first_newline > 0:
                end_content = end_content[first_newline+1:]
            end_lines = end_content.splitlines()
            end_start_line = len(lines) - len(end_lines) + 1
            
            # Build numbered snippet
            begin_numbered = add_line_numbers('\n'.join(begin_lines), 1)
            end_numbered = add_line_numbers('\n'.join(end_lines), end_start_line)
            
            omitted = len(lines) - len(begin_lines) - len(end_lines)
            truncation_note = f"\n     | ... ({omitted} lines omitted, lines {len(begin_lines)+1}-{end_start_line-1}) ...\n"
            
            numbered_content = begin_numbered + truncation_note + end_numbered
        else:
            numbered_content = add_line_numbers(content, 1)
        
        snippets.append(f"--- FILE: {path} ({len(lines)} lines) ---\n{numbered_content}")
    
    return "\n\n".join(snippets), line_counts


def parse_line_patches(llm_response: str) -> List[Dict[str, Any]]:
    """Parse line-based patches from LLM response.
    
    Expected JSON format:
    {
        "patches": [
            {
                "path": "src/file.ts",
                "changes": [
                    {
                        "lines": [10, 15],  // start and end line (inclusive)
                        "new_content": "replacement code here",
                        "description": "what this change does"
                    }
                ],
                "add_at_line": 1,  // optional: insert new content at line
                "add_content": "import { X } from 'y';"  // content to insert
            }
        ]
    }
    
    Args:
        llm_response: Raw LLM response
        
    Returns:
        List of patch dicts with line info
    """
    patches = []
    
    # Try to extract JSON
    try:
        # Find JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_response)
        if json_match:
            import json
            data = json.loads(json_match.group(1))
        else:
            # Try raw JSON
            import json
            json_start = llm_response.find('{')
            if json_start >= 0:
                data = json.loads(llm_response[json_start:])
            else:
                return []
        
        # Extract patches
        raw_patches = data.get("patches", data.get("changes", []))
        for p in raw_patches:
            if isinstance(p, dict) and p.get("path"):
                patches.append(p)
                # Log what we parsed for debugging
                changes = p.get("changes", [])
                logger.debug(f"Parsed patch for {p.get('path')}: {len(changes)} changes")
                for i, c in enumerate(changes):
                    lines = c.get("lines", [])
                    content = c.get("new_content", c.get("content", ""))
                    # Coerce list-like content to string if LLM returned an array of lines
                    if isinstance(content, list):
                        try:
                            content = "\n".join(str(x) for x in content)
                        except Exception:
                            content = str(content)
                    content_preview = (content[:50].replace('\n', '\\n') if isinstance(content, str) and content else "(EMPTY)")
                    logger.debug(f"  Change {i+1}: lines {lines}, content: {content_preview}...")
                
    except Exception as e:
        logger.warning(f"Failed to parse line patches: {e}")
    
    return patches


def apply_line_patches(
    original: str,
    patches: List[Dict[str, Any]],
) -> LinePatchResult:
    """Apply line-based patches to file content.
    
    Args:
        original: Original file content
        patches: List of patch operations for this file
        
    Returns:
        LinePatchResult with patched content
    """
    lines = original.splitlines()
    total_lines = len(lines)
    
    result = LinePatchResult(
        content=original,
        success=True,
        applied_count=0,
        total_count=0,
    )
    
    # Collect all changes (for this file)
    changes = []
    for p in patches:
        # Handle inline changes array
        for change in p.get("changes", []):
            line_range = change.get("lines", [])
            if len(line_range) >= 2:
                new_content = change.get("new_content", change.get("content", ""))
                # Coerce list-like new_content to string
                if isinstance(new_content, list):
                    try:
                        new_content = "\n".join(str(x) for x in new_content)
                    except Exception:
                        new_content = str(new_content)

                # Warn if replacement is empty (potential LLM mistake)
                if not new_content or (isinstance(new_content, str) and not new_content.strip()):
                    range_str = f"{line_range[0]}-{line_range[1]}"
                    lines_being_deleted = line_range[1] - line_range[0] + 1
                    logger.warning(
                        f"Empty replacement for lines {range_str} ({lines_being_deleted} lines) - "
                        f"this will DELETE the code. Description: {change.get('description', 'none')}"
                    )
                    # Skip empty replacements unless explicitly marked as delete
                    if change.get("operation") != "delete" and lines_being_deleted > 1:
                        logger.warning(f"Skipping suspicious empty replacement for {range_str}")
                        result.warnings.append(f"Skipped empty replacement for lines {range_str}")
                        continue
                
                changes.append({
                    "start": line_range[0],
                    "end": line_range[1],
                    "content": new_content,
                    "description": change.get("description", ""),
                    "operation": "replace",
                })
        
        # Handle add_at_line for insertions
        if p.get("add_at_line") and p.get("add_content"):
            add_content = p.get("add_content")
            if isinstance(add_content, list):
                try:
                    add_content = "\n".join(str(x) for x in add_content)
                except Exception:
                    add_content = str(add_content)
            changes.append({
                "start": p["add_at_line"],
                "end": p["add_at_line"],
                "content": add_content,
                "description": p.get("add_description", "Insert new content"),
                "operation": "insert",
            })
    
    result.total_count = len(changes)
    
    if not changes:
        return result
    
    # Sort changes by line number (descending) to apply from bottom up
    # This prevents line number shifts from affecting subsequent patches
    changes.sort(key=lambda x: x["start"], reverse=True)
    
    # Apply each change
    for change in changes:
        start = change["start"]
        end = change["end"]
        new_content = change["content"]
        operation = change.get("operation", "replace")
        
        # Validate line numbers
        if start < 1:
            result.errors.append(f"Invalid start line {start} (must be >= 1)")
            continue
        if end < start:
            result.errors.append(f"Invalid line range {start}-{end} (end < start)")
            continue
        if start > total_lines + 1:
            result.warnings.append(f"Start line {start} beyond file end ({total_lines}), appending")
            start = total_lines + 1
            end = start
        if end > total_lines:
            result.warnings.append(f"End line {end} beyond file end ({total_lines}), adjusting")
            end = total_lines
        
        # Apply the change
        if operation == "insert":
            # Insert before the specified line
            new_lines = new_content.splitlines()
            lines = lines[:start-1] + new_lines + lines[start-1:]
            total_lines = len(lines)
            result.applied_count += 1
            logger.debug(f"Inserted {len(new_lines)} lines at line {start}")
        elif operation == "delete":
            # Delete the line range
            lines = lines[:start-1] + lines[end:]
            total_lines = len(lines)
            result.applied_count += 1
            logger.debug(f"Deleted lines {start}-{end}")
        else:  # replace
            # Replace line range with new content
            new_lines = new_content.splitlines()
            lines = lines[:start-1] + new_lines + lines[end:]
            total_lines = len(lines)
            result.applied_count += 1
            logger.debug(f"Replaced lines {start}-{end} with {len(new_lines)} lines")
    
    result.content = "\n".join(lines)
    result.success = result.applied_count > 0 and len(result.errors) == 0
    
    return result


def convert_search_replace_to_line_patch(
    search: str,
    replace: str,
    original_content: str,
) -> Optional[Dict[str, Any]]:
    """Convert a search/replace operation to line-based patch.
    
    Finds the search text in the original and returns line numbers.
    
    Args:
        search: Text to search for
        replace: Replacement text
        original_content: Original file content
        
    Returns:
        Dict with lines and new_content, or None if not found
    """
    if not search or search not in original_content:
        return None
    
    lines = original_content.splitlines()
    
    # Find the start position
    pos = original_content.find(search)
    if pos < 0:
        return None
    
    # Convert position to line number
    content_before = original_content[:pos]
    start_line = content_before.count('\n') + 1
    
    # Find end line
    search_line_count = search.count('\n') + 1
    end_line = start_line + search_line_count - 1
    
    return {
        "lines": [start_line, end_line],
        "new_content": replace,
        "description": f"Replace lines {start_line}-{end_line}",
    }
