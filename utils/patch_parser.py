"""Patch parsing utilities for extracting code patches from LLM responses.

This module provides functions to parse various LLM output formats:
- JSON format with patches array
- Marker format (--- FILE: path ---)
- SEARCH/REPLACE block format
- Mixed and malformed formats

Extracted from refactor_agent.py for better separation of concerns.
"""

from typing import Dict, Any, List, Optional
import json
import re

from utils.logging import get_logger

logger = get_logger("patch_parser")


def parse_json_patches(text: str) -> List[Dict[str, str]]:
    """Parse patches from JSON format in LLM response.
    
    Handles various LLM output formats:
    1. Clean JSON
    2. JSON wrapped in markdown code blocks
    3. JSON with embedded code blocks for file content
    
    Args:
        text: Raw LLM response text
        
    Returns:
        List of patch dicts with 'path' and 'patched' keys
    """
    # Step 1: Strip markdown code block wrapper if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove trailing ```
    text = text.strip()
    
    # Step 2: Try direct JSON parsing first
    try:
        data = json.loads(text)
        return extract_patches_from_data(data)
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try to find JSON object with regex
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return extract_patches_from_data(data)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed: {e}")
    
    # Step 4: Try to extract patches with embedded code blocks
    patches = parse_json_with_code_blocks(text)
    if patches:
        return patches
    
    # Step 5: Try a more lenient extraction - find patches array directly
    patches = extract_patches_lenient(text)
    if patches:
        return patches
    
    logger.debug("All JSON parsing strategies failed")
    return []


def extract_patches_from_data(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract patches from parsed JSON data.
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        List of patch dicts with 'path' and 'patched' keys
    """
    patches = []
    raw_patches = data.get("patches", [])
    
    if not raw_patches:
        logger.debug("JSON parsed but no 'patches' array found")
        return []
    
    for p in raw_patches:
        path = p.get("path")
        patched = p.get("patched")
        
        if not path:
            logger.warning("Patch missing 'path' field")
            continue
        if not patched:
            logger.warning(f"Patch for {path} missing 'patched' content")
            continue
        
        # Clean up the patched content (remove code block markers if present)
        patched = clean_code_content(patched)
            
        patches.append({
            "path": path,
            "patched": patched
        })
    
    return patches


def clean_code_content(content: str) -> str:
    """Remove code block markers from content.
    
    Args:
        content: Code content possibly wrapped in markdown code blocks
        
    Returns:
        Clean code content without markers
    """
    content = content.strip()
    # Remove leading code block marker with optional language
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        else:
            content = content[3:]
    # Remove trailing code block marker
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def parse_json_with_code_blocks(text: str) -> List[Dict[str, str]]:
    """Parse JSON where patched content might have embedded code blocks.
    
    Some LLMs output:
    {
      "patches": [
        {
          "path": "file.cs",
          "patched": "```csharp
          ...actual code...
          ```"
        }
      ]
    }
    
    Args:
        text: Raw LLM response text
        
    Returns:
        List of patch dicts
    """
    patches = []
    
    # Find all path/patched pairs using a more flexible pattern
    path_pattern = r'"path"\s*:\s*"([^"]+)"'
    
    # Find all paths first
    path_matches = list(re.finditer(path_pattern, text))
    
    for i, path_match in enumerate(path_matches):
        path = path_match.group(1)
        
        # Find the "patched" field after this path
        start_search = path_match.end()
        end_search = path_matches[i + 1].start() if i + 1 < len(path_matches) else len(text)
        
        section = text[start_search:end_search]
        
        # Look for patched content - might be a code block
        patched_match = re.search(r'"patched"\s*:\s*"', section)
        if patched_match:
            # Find where the value starts
            value_start = patched_match.end()
            content_section = section[value_start:]
            
            # Try to find the end by looking for unescaped quote followed by comma/bracket
            patched_content = extract_json_string_value(content_section)
            if patched_content:
                patched_content = clean_code_content(patched_content)
                patches.append({"path": path, "patched": patched_content})
    
    return patches


def extract_json_string_value(text: str) -> Optional[str]:
    """Extract a JSON string value, handling escaped characters.
    
    Args:
        text: Text starting at the beginning of a JSON string value (after opening quote)
        
    Returns:
        Extracted and unescaped string value, or None if parsing fails
    """
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == '\\' and i + 1 < len(text):
            # Escaped character
            next_char = text[i + 1]
            if next_char == 'n':
                result.append('\n')
            elif next_char == 't':
                result.append('\t')
            elif next_char == 'r':
                result.append('\r')
            elif next_char == '"':
                result.append('"')
            elif next_char == '\\':
                result.append('\\')
            else:
                result.append(next_char)
            i += 2
        elif char == '"':
            # End of string
            return ''.join(result)
        else:
            result.append(char)
            i += 1
    
    # Didn't find closing quote - return what we have
    return ''.join(result) if result else None


def extract_patches_lenient(text: str) -> List[Dict[str, str]]:
    """Lenient extraction when JSON parsing fails.
    
    Looks for file paths and code content directly.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        List of patch dicts
    """
    patches = []
    
    # Find patterns like: "path": "...", "patched": <content>
    path_matches = re.findall(r'"path"\s*:\s*"([^"]+)"', text)
    
    for path in path_matches:
        # Try to find associated code block after the path mention
        path_pos = text.find(f'"path": "{path}"') 
        if path_pos == -1:
            path_pos = text.find(f'"path":"{path}"')
        if path_pos == -1:
            continue
        
        # Look for code block after this path
        remaining = text[path_pos:]
        
        # Look for code block markers
        code_block_match = re.search(
            r'```(?:csharp|cs|python|java|javascript|typescript)?\n([\s\S]*?)```', 
            remaining
        )
        if code_block_match:
            patches.append({
                "path": path,
                "patched": code_block_match.group(1).strip()
            })
    
    return patches


def parse_marker_patches(text: str) -> List[Dict[str, str]]:
    """Parse file marker format, handling both full content and SEARCH/REPLACE blocks.
    
    Format:
        --- FILE: path/to/file.cs ---
        <content or SEARCH/REPLACE blocks>
    
    Args:
        text: Raw LLM response text
        
    Returns:
        List of patch dicts with 'path' and either 'patched' or 'search_replace_blocks'
    """
    pattern = r"--- FILE: (.+?) ---\n"
    parts = re.split(pattern, text)
    patches = []
    if len(parts) < 3:
        return patches
    
    # drop leading preamble
    it = iter(parts[1:])
    for path, content in zip(it, it):
        path = path.strip()
        content = content.strip()
        
        # Check if content contains SEARCH/REPLACE blocks
        if "<<<<<<< SEARCH" in content and ">>>>>>> REPLACE" in content:
            patches.append({
                "path": path,
                "search_replace_blocks": content,  # Will be processed later
                "patched": None  # Indicates we need to apply search/replace
            })
        else:
            patches.append({"path": path, "patched": content})
    
    return patches


def parse_search_replace_json(text: str) -> List[Dict[str, Any]]:
    """Parse JSON format with search_replace changes instead of full patches.
    
    Format:
        {
          "changes": [
            {
              "path": "file.cs",
              "search_replace": [
                {"search": "old code", "replace": "new code"}
              ]
            }
          ]
        }
    
    Args:
        text: Raw LLM response text
        
    Returns:
        List of dicts with 'path' and 'search_replace' keys
    """
    try:
        # Strip markdown wrapper
        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            text = text[first_newline + 1:] if first_newline != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Try to parse JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return []
        
        data = json.loads(json_match.group())
        changes = data.get("changes", [])
        
        if not changes:
            return []
        
        result = []
        for change in changes:
            path = change.get("path")
            search_replace = change.get("search_replace", [])
            if path and search_replace:
                result.append({
                    "path": path,
                    "search_replace": search_replace
                })
        
        return result
    except Exception as e:
        logger.debug(f"Failed to parse search_replace JSON: {e}")
        return []


def infer_patches(llm_response: Any, cycle_files: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Parse patches from LLM response using multiple strategies.
    
    This is the main entry point for patch parsing. It tries multiple
    parsing strategies in order of preference.
    
    Args:
        llm_response: Raw LLM response (string or dict)
        cycle_files: Optional list of cycle files for context (unused currently)
    
    Returns:
        List of dicts with either:
        - {"path": str, "patched": str} for full content patches
        - {"path": str, "search_replace_blocks": str} for marker-style S/R
        - {"path": str, "search_replace": List[Dict]} for JSON-style S/R
    """
    import json as json_module
    text = llm_response if isinstance(llm_response, str) else json_module.dumps(llm_response)

    # Try structured JSON first (handles both full patches and search_replace format)
    patches = parse_json_patches(text)
    if patches:
        logger.debug(f"Parsed {len(patches)} patches from JSON format")
        return patches

    # Try JSON with search_replace changes
    patches = parse_search_replace_json(text)
    if patches:
        logger.debug(f"Parsed {len(patches)} search/replace patches from JSON format")
        return patches

    # Try marker format (handles both full content and SEARCH/REPLACE blocks)
    patches = parse_marker_patches(text)
    if patches:
        logger.debug(f"Parsed {len(patches)} patches from marker format")
        return patches

    # Log the first part of response to help debug parsing issues
    logger.warning(f"Could not parse patches from LLM response. First 500 chars: {text[:500]}")
    
    # Nothing parsed: return empty => no-op
    return []
