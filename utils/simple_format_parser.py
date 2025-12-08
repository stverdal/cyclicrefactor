"""Parser for simple line-based LLM output format.

This module parses the simplified output format designed for smaller LLMs
that may struggle with JSON. The format uses clear delimiters and is more
forgiving of minor formatting issues.

Format example:
```
STRATEGY: interface_extraction

NEW_FILE: Interfaces/IFoo.cs
---BEGIN---
<file content>
---END---

MODIFY: path/to/file.cs
ADD_USING: using Some.Namespace;
REMOVE_USING: using Other.Namespace;
SEARCH:
<exact search text>
REPLACE:
<replacement text>

CYCLE_BROKEN: Yes - explanation
```
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("simple_format_parser")


@dataclass
class SimpleFormatResult:
    """Result of parsing simple format output."""
    success: bool = False
    strategy: str = ""
    new_files: List[Dict[str, str]] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    cycle_broken: str = ""
    parse_errors: List[str] = field(default_factory=list)
    raw_response: str = ""


def parse_simple_format(response: str) -> SimpleFormatResult:
    """Parse the simple line-based format from LLM response.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        SimpleFormatResult with parsed data
    """
    result = SimpleFormatResult(raw_response=response)
    
    # Clean up response - remove markdown code blocks if present
    response = _clean_response(response)
    
    # Extract strategy
    strategy_match = re.search(r'^STRATEGY:\s*(\S+)', response, re.MULTILINE | re.IGNORECASE)
    if strategy_match:
        result.strategy = strategy_match.group(1).strip()
    else:
        result.parse_errors.append("Missing STRATEGY line")
    
    # Extract new files
    result.new_files = _extract_new_files(response)
    
    # Extract modifications
    result.modifications = _extract_modifications(response)
    
    # Extract cycle_broken status
    cycle_match = re.search(r'^CYCLE_BROKEN:\s*(.+?)(?=\n(?:NEW_FILE|MODIFY|$)|\Z)', 
                           response, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if cycle_match:
        result.cycle_broken = cycle_match.group(1).strip()
    
    # Determine success
    result.success = (
        len(result.parse_errors) == 0 and
        result.strategy != "" and
        (len(result.new_files) > 0 or len(result.modifications) > 0)
    )
    
    if result.success:
        logger.info(f"Parsed simple format: strategy={result.strategy}, "
                   f"new_files={len(result.new_files)}, mods={len(result.modifications)}")
    else:
        logger.warning(f"Simple format parse issues: {result.parse_errors}")
    
    return result


def _clean_response(response: str) -> str:
    """Remove markdown code blocks and clean up response."""
    # Remove ```...``` blocks
    response = re.sub(r'^```\w*\n', '', response, flags=re.MULTILINE)
    response = re.sub(r'\n```$', '', response, flags=re.MULTILINE)
    response = response.replace('```', '')
    return response.strip()


def _extract_new_files(response: str) -> List[Dict[str, str]]:
    """Extract NEW_FILE blocks from response."""
    new_files = []
    
    # Pattern: NEW_FILE: path\n---BEGIN---\ncontent\n---END---
    pattern = r'NEW_FILE:\s*(.+?)\n---BEGIN---\n(.*?)\n---END---'
    matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        path = match.group(1).strip()
        content = match.group(2).strip()
        
        if path and content:
            new_files.append({
                "path": path,
                "content": content
            })
            logger.debug(f"Extracted new file: {path} ({len(content)} chars)")
    
    return new_files


def _extract_modifications(response: str) -> List[Dict[str, Any]]:
    """Extract MODIFY blocks from response."""
    modifications = []
    
    # Split by MODIFY: to get each modification block
    parts = re.split(r'\nMODIFY:\s*', response, flags=re.IGNORECASE)
    
    for part in parts[1:]:  # Skip first part (before first MODIFY)
        mod = _parse_modification_block(part)
        if mod:
            modifications.append(mod)
    
    return modifications


def _parse_modification_block(block: str) -> Optional[Dict[str, Any]]:
    """Parse a single MODIFY block."""
    lines = block.split('\n')
    if not lines:
        return None
    
    # First line is the file path
    path = lines[0].strip()
    if not path:
        return None
    
    mod = {
        "path": path,
        "add_using": [],
        "remove_using": [],
        "search_replace": [],
        "after_class": None
    }
    
    block_text = '\n'.join(lines[1:])
    
    # Extract ADD_USING
    for match in re.finditer(r'^ADD_USING:\s*(.+)$', block_text, re.MULTILINE | re.IGNORECASE):
        using = match.group(1).strip()
        if using:
            mod["add_using"].append(using)
    
    # Extract REMOVE_USING
    for match in re.finditer(r'^REMOVE_USING:\s*(.+)$', block_text, re.MULTILINE | re.IGNORECASE):
        using = match.group(1).strip()
        if using:
            mod["remove_using"].append(using)
    
    # Extract AFTER_CLASS_DECLARATION
    class_match = re.search(r'^AFTER_CLASS_DECLARATION:\s*(.+)$', block_text, re.MULTILINE | re.IGNORECASE)
    if class_match:
        mod["after_class"] = class_match.group(1).strip()
    
    # Extract SEARCH/REPLACE pairs
    search_replace_pattern = r'SEARCH:\n(.*?)\nREPLACE:\n(.*?)(?=\nSEARCH:|\nADD_USING:|\nREMOVE_USING:|\nCYCLE_BROKEN:|\nMODIFY:|\Z)'
    for match in re.finditer(search_replace_pattern, block_text, re.DOTALL | re.IGNORECASE):
        search = match.group(1).strip()
        replace = match.group(2).strip()
        
        if search:  # Only add if search is non-empty
            mod["search_replace"].append({
                "search": search,
                "replace": replace
            })
    
    return mod


def convert_to_standard_format(simple_result: SimpleFormatResult) -> Dict[str, Any]:
    """Convert SimpleFormatResult to standard JSON format used by the pipeline.
    
    Args:
        simple_result: Parsed simple format result
        
    Returns:
        Dictionary in standard refactor output format
    """
    output = {
        "reasoning": f"Strategy: {simple_result.strategy}. {simple_result.cycle_broken}",
        "strategy_used": simple_result.strategy,
        "new_files": simple_result.new_files,
        "changes": []
    }
    
    for mod in simple_result.modifications:
        change = {
            "path": mod["path"],
            "search_replace": mod["search_replace"]
        }
        
        # Add prepend for ADD_USING
        if mod["add_using"]:
            change["prepend"] = "\n".join(mod["add_using"])
        
        # Handle AFTER_CLASS_DECLARATION specially
        if mod["after_class"]:
            # This needs to be converted to a search/replace for class declaration
            change["class_modification"] = mod["after_class"]
        
        output["changes"].append(change)
    
    return output


def is_simple_format(response: str) -> bool:
    """Check if response appears to be in simple format vs JSON.
    
    Args:
        response: Raw LLM response
        
    Returns:
        True if response looks like simple format, False if looks like JSON
    """
    # Check for simple format markers
    has_strategy = bool(re.search(r'^STRATEGY:', response, re.MULTILINE | re.IGNORECASE))
    has_new_file = bool(re.search(r'^NEW_FILE:', response, re.MULTILINE | re.IGNORECASE))
    has_modify = bool(re.search(r'^MODIFY:', response, re.MULTILINE | re.IGNORECASE))
    
    # Check for JSON markers
    looks_like_json = response.strip().startswith('{') or '{"reasoning"' in response
    
    return (has_strategy or has_new_file or has_modify) and not looks_like_json
