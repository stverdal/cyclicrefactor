"""Simple format mode for smaller LLMs (7B-14B parameters).

This module provides a simpler text-based output format that's easier for
smaller models to generate correctly compared to complex JSON structures.

Benefits:
- No nested JSON to get wrong
- Line-based parsing is more forgiving
- Clear visual markers
- Works with models that struggle with brackets/braces
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("simple_format")


@dataclass
class SimpleChange:
    """A single change parsed from simple format."""
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    new_content: str = ""
    description: str = ""
    is_new_file: bool = False
    operation: str = "replace"  # replace, insert, delete, append


@dataclass
class SimpleFormatResult:
    """Result of parsing simple format output."""
    strategy: str = ""
    reasoning: str = ""
    changes: List[SimpleChange] = field(default_factory=list)
    raw_text: str = ""
    parse_warnings: List[str] = field(default_factory=list)


def build_simple_format_prompt(
    cycle_info: Dict[str, Any],
    file_snippets: str,
    strategy_hint: Optional[str] = None,
    pattern_example: str = "",
) -> str:
    """Build a prompt that asks for simple text format output.
    
    Args:
        cycle_info: Cycle specification dict
        file_snippets: Numbered file content
        strategy_hint: Optional strategy from describer
        pattern_example: Optional pattern example
        
    Returns:
        Prompt text requesting simple format output
    """
    cycle_id = cycle_info.get("id", "unknown")
    nodes = cycle_info.get("graph", {}).get("nodes", [])
    edges = cycle_info.get("graph", {}).get("edges", [])
    
    # Format edges simply
    edge_strs = []
    for e in edges[:10]:  # Limit to avoid huge prompts
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edge_strs.append(f"  {e[0]} -> {e[1]}")
    edges_text = "\n".join(edge_strs) if edge_strs else "  (see file imports)"
    
    prompt = f"""Break this cyclic dependency by modifying the code.

CYCLE: {cycle_id}
NODES: {', '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}
EDGES:
{edges_text}

STRATEGY HINT: {strategy_hint or 'Choose the best approach'}

{file_snippets}

=== OUTPUT FORMAT ===

Use this EXACT format (copy the markers exactly):

STRATEGY: <name of strategy used>
REASONING: <one sentence explaining the approach>

For each file change:

FILE: <path/to/file.ext>
CHANGE: lines <start>-<end>
---
<new code to replace those lines>
---
DESCRIPTION: <what this change does>

For new files:

NEW_FILE: <path/to/new/file.ext>
---
<complete file content>
---

=== EXAMPLE ===

STRATEGY: interface_extraction
REASONING: Extract interface to break bidirectional dependency

FILE: src/auth/AuthService.ts
CHANGE: lines 1-2
---
import {{ IUserProvider }} from '../shared/IUserProvider';
---
DESCRIPTION: Import interface instead of concrete class

FILE: src/auth/AuthService.ts
CHANGE: lines 15-17
---
  constructor(private userProvider: IUserProvider) {{
    this.logger = new Logger();
  }}
---
DESCRIPTION: Use interface type in constructor

NEW_FILE: src/shared/IUserProvider.ts
---
export interface IUserProvider {{
  getUserById(id: string): Promise<User>;
  getCurrentUser(): User | null;
}}
---

=== RULES ===
1. Use EXACT line numbers from the file snippets above
2. Put code between --- markers (three dashes)
3. Include complete replacement code (not just the changed parts)
4. Keep indentation consistent
5. One FILE/CHANGE block per change location

Now generate the changes:
"""
    return prompt


def parse_simple_format(text: str) -> SimpleFormatResult:
    """Parse LLM output in simple text format.
    
    This parser is intentionally forgiving of minor format variations.
    
    Args:
        text: Raw LLM output text
        
    Returns:
        SimpleFormatResult with parsed changes
    """
    result = SimpleFormatResult(raw_text=text)
    
    # Extract strategy
    strategy_match = re.search(r'^STRATEGY:\s*(.+?)$', text, re.MULTILINE | re.IGNORECASE)
    if strategy_match:
        result.strategy = strategy_match.group(1).strip()
    
    # Extract reasoning
    reasoning_match = re.search(r'^REASONING:\s*(.+?)$', text, re.MULTILINE | re.IGNORECASE)
    if reasoning_match:
        result.reasoning = reasoning_match.group(1).strip()
    
    # Parse FILE/CHANGE blocks
    # Pattern: FILE: path\nCHANGE: lines X-Y\n---\ncontent\n---\nDESCRIPTION: desc
    file_pattern = re.compile(
        r'^FILE:\s*(.+?)\s*$\s*'
        r'^CHANGE:\s*(?:lines?\s*)?(\d+)\s*[-–—to]+\s*(\d+)\s*$\s*'
        r'^---\s*$\s*'
        r'(.*?)'
        r'^---\s*$\s*'
        r'(?:^DESCRIPTION:\s*(.+?)\s*$)?',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    for match in file_pattern.finditer(text):
        file_path = match.group(1).strip()
        start_line = int(match.group(2))
        end_line = int(match.group(3))
        content = match.group(4).strip()
        description = match.group(5).strip() if match.group(5) else ""
        
        result.changes.append(SimpleChange(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            new_content=content,
            description=description,
            is_new_file=False,
        ))
    
    # Also try simpler pattern without DESCRIPTION
    simple_file_pattern = re.compile(
        r'^FILE:\s*(.+?)\s*$\s*'
        r'^CHANGE:\s*(?:lines?\s*)?(\d+)\s*[-–—to]+\s*(\d+)\s*$\s*'
        r'^---\s*$\s*'
        r'(.*?)'
        r'^---\s*$',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    # Find matches not already captured
    existing_positions = {(c.file_path, c.start_line, c.end_line) for c in result.changes}
    
    for match in simple_file_pattern.finditer(text):
        file_path = match.group(1).strip()
        start_line = int(match.group(2))
        end_line = int(match.group(3))
        
        if (file_path, start_line, end_line) not in existing_positions:
            content = match.group(4).strip()
            result.changes.append(SimpleChange(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                new_content=content,
                is_new_file=False,
            ))
    
    # Parse NEW_FILE blocks
    new_file_pattern = re.compile(
        r'^NEW_FILE:\s*(.+?)\s*$\s*'
        r'^---\s*$\s*'
        r'(.*?)'
        r'^---\s*$',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    for match in new_file_pattern.finditer(text):
        file_path = match.group(1).strip()
        content = match.group(2).strip()
        
        result.changes.append(SimpleChange(
            file_path=file_path,
            new_content=content,
            is_new_file=True,
        ))
    
    # Try alternate patterns if we got nothing
    if not result.changes:
        result = _try_alternate_patterns(text, result)
    
    # Log parse results
    if result.changes:
        logger.info(f"Parsed {len(result.changes)} changes from simple format "
                   f"(strategy: {result.strategy})")
    else:
        logger.warning("No changes parsed from simple format output")
        result.parse_warnings.append("Could not parse any changes from output")
    
    return result


def _try_alternate_patterns(text: str, result: SimpleFormatResult) -> SimpleFormatResult:
    """Try alternate patterns for more lenient parsing.
    
    Handles common variations like:
    - Different dash styles (-, –, —)
    - Missing CHANGE keyword
    - Line numbers in different formats
    """
    # Pattern: FILE: path (lines X-Y)\n---\ncontent\n---
    alt_pattern = re.compile(
        r'^FILE:\s*(.+?)\s*\((?:lines?\s*)?(\d+)\s*[-–—to]+\s*(\d+)\)\s*$\s*'
        r'^---\s*$\s*'
        r'(.*?)'
        r'^---\s*$',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    for match in alt_pattern.finditer(text):
        result.changes.append(SimpleChange(
            file_path=match.group(1).strip(),
            start_line=int(match.group(2)),
            end_line=int(match.group(3)),
            new_content=match.group(4).strip(),
            is_new_file=False,
        ))
    
    # Pattern: Just code blocks with file paths
    # ```typescript\n// FILE: path\ncontent\n```
    code_block_pattern = re.compile(
        r'```\w*\s*\n'
        r'(?://|#|/\*)\s*FILE:\s*(.+?)\s*(?:\*/)?(?:\s*\n|\s+)'
        r'(.*?)'
        r'```',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    
    for match in code_block_pattern.finditer(text):
        file_path = match.group(1).strip()
        content = match.group(2).strip()
        # Check if it looks like a complete new file
        if 'export ' in content or 'import ' in content or 'class ' in content:
            result.changes.append(SimpleChange(
                file_path=file_path,
                new_content=content,
                is_new_file=True,  # Assume new file if full content
            ))
            result.parse_warnings.append(
                f"Parsed {file_path} from code block (assumed new file)"
            )
    
    return result


def convert_simple_to_line_patches(
    simple_result: SimpleFormatResult,
) -> List[Dict[str, Any]]:
    """Convert simple format result to line-patch format.
    
    This allows reusing the line_patch.apply_line_patches function.
    
    Args:
        simple_result: Parsed simple format result
        
    Returns:
        List of patch dicts in line_patch format
    """
    patches_by_file: Dict[str, Dict[str, Any]] = {}
    
    for change in simple_result.changes:
        if change.is_new_file:
            # New files are handled separately
            patches_by_file[change.file_path] = {
                "path": change.file_path,
                "is_new_file": True,
                "content": change.new_content,
            }
        else:
            # Accumulate changes per file
            if change.file_path not in patches_by_file:
                patches_by_file[change.file_path] = {
                    "path": change.file_path,
                    "changes": [],
                }
            
            patches_by_file[change.file_path]["changes"].append({
                "lines": [change.start_line, change.end_line],
                "new_content": change.new_content,
                "description": change.description,
            })
    
    return list(patches_by_file.values())


def convert_simple_to_search_replace(
    simple_result: SimpleFormatResult,
    file_contents: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Convert simple format to search/replace format.
    
    Uses line numbers to extract the original text for SEARCH blocks.
    
    Args:
        simple_result: Parsed simple format result
        file_contents: Dict mapping file paths to their content
        
    Returns:
        List of patch dicts with search_replace operations
    """
    patches = []
    
    for change in simple_result.changes:
        if change.is_new_file:
            patches.append({
                "path": change.file_path,
                "patched": change.new_content,
                "is_new_file": True,
            })
        else:
            # Find file content (try exact and basename match)
            content = file_contents.get(change.file_path)
            if not content:
                basename = change.file_path.split('/')[-1].split('\\')[-1]
                for path, c in file_contents.items():
                    if path.endswith(basename):
                        content = c
                        break
            
            if not content:
                logger.warning(f"No content found for {change.file_path}")
                continue
            
            # Extract original lines
            lines = content.splitlines()
            start = max(0, (change.start_line or 1) - 1)
            end = min(len(lines), change.end_line or len(lines))
            
            original_text = "\n".join(lines[start:end])
            
            # Find or create patch entry for this file
            patch_entry = None
            for p in patches:
                if p.get("path") == change.file_path and not p.get("is_new_file"):
                    patch_entry = p
                    break
            
            if not patch_entry:
                patch_entry = {
                    "path": change.file_path,
                    "search_replace": [],
                }
                patches.append(patch_entry)
            
            patch_entry["search_replace"].append({
                "search": original_text,
                "replace": change.new_content,
            })
    
    return patches


# Model size detection for auto-enabling
def should_use_simple_format(
    model_name: str,
    auto_threshold_b: int = 14,
    force_simple: bool = False,
) -> bool:
    """Determine if simple format should be used based on model size.
    
    Args:
        model_name: Name of the model (e.g., "qwen2.5-coder:7b")
        auto_threshold_b: Auto-enable for models <= this size in billions
        force_simple: If True, always use simple format
        
    Returns:
        True if simple format should be used
    """
    if force_simple:
        return True
    
    # Extract size from model name
    model_lower = model_name.lower()
    
    # Common patterns: 7b, 13b, 14b, 32b, 70b
    size_match = re.search(r'(\d+)b', model_lower)
    if size_match:
        size_b = int(size_match.group(1))
        if size_b <= auto_threshold_b:
            logger.info(f"Auto-enabling simple format for {size_b}B model")
            return True
    
    # Check for known small model names
    small_models = ['phi', 'gemma:2b', 'tinyllama', 'stablelm']
    for small in small_models:
        if small in model_lower:
            logger.info(f"Auto-enabling simple format for small model: {model_name}")
            return True
    
    return False
