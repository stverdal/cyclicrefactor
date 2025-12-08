"""Utility for selecting relevant snippets from source files.

This module provides language-aware heuristics to extract the most relevant
portions of source files for inclusion in LLM prompts, helping to stay within
token limits while preserving important context.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional


def prioritize_full_files(
    files: List[Dict[str, Any]],
    total_budget_chars: int,
    max_chars_per_file: int = 8000,
    budget_pct: float = 0.4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Prioritize including full file content for smaller files that fit in budget.
    
    This helps LLM see complete context instead of snippets, improving accuracy.
    Files are sorted by size (smallest first) to maximize the number of complete files.
    
    Args:
        files: List of file dicts with 'path' and 'content' keys
        total_budget_chars: Total character budget for all file content
        max_chars_per_file: Maximum chars per file to consider "small enough" for full inclusion
        budget_pct: Maximum percentage of budget to use for full files
        
    Returns:
        Tuple of (full_files, remaining_files) where full_files should be included
        completely and remaining_files need snippet extraction
    """
    if not files:
        return [], []
    
    # Calculate the budget for full-file inclusion
    full_file_budget = int(total_budget_chars * budget_pct)
    
    # Create list with sizes
    sized_files = []
    for f in files:
        content = f.get('content', '') or ''
        size = len(content)
        sized_files.append({
            'file': f,
            'size': size,
            'fits': size <= max_chars_per_file and size > 0
        })
    
    # Sort by size (smallest first) to maximize number of full files
    sized_files.sort(key=lambda x: x['size'])
    
    full_files = []
    remaining_files = []
    used_budget = 0
    
    for item in sized_files:
        f = item['file']
        size = item['size']
        
        # Check if this file fits in remaining budget AND is under per-file limit
        if item['fits'] and (used_budget + size) <= full_file_budget:
            full_files.append(f)
            used_budget += size
        else:
            remaining_files.append(f)
    
    return full_files, remaining_files


def build_file_snippets_with_priority(
    files: List[Dict[str, Any]],
    cycle: Dict[str, Any],
    total_budget_chars: int,
    prioritize_full: bool = True,
    max_chars_per_file: int = 8000,
    full_file_budget_pct: float = 0.4,
    snippet_max_chars: int = 4000,
) -> Tuple[str, Dict[str, str]]:
    """Build file snippets with priority for full file inclusion.
    
    Args:
        files: List of file dicts with 'path' and 'content' keys
        cycle: Cycle spec dict for snippet extraction
        total_budget_chars: Total character budget
        prioritize_full: If True, include full content for small files
        max_chars_per_file: Max chars for "small" file classification
        full_file_budget_pct: Percentage of budget for full files
        snippet_max_chars: Max chars per snippet for larger files
        
    Returns:
        Tuple of (combined_snippets_string, file_status_dict) where status shows
        whether each file was included in full or as snippet
    """
    file_status = {}
    snippets = []
    used_chars = 0
    
    if prioritize_full:
        full_files, remaining_files = prioritize_full_files(
            files, total_budget_chars, max_chars_per_file, full_file_budget_pct
        )
        
        # Add full files first
        for f in full_files:
            path = f.get('path', 'unknown')
            content = f.get('content', '') or ''
            snippets.append(f"--- FILE: {path} (COMPLETE) ---\n{content}")
            file_status[path] = 'full'
            used_chars += len(content)
        
        # Calculate remaining budget for snippets
        remaining_budget = total_budget_chars - used_chars
        per_file_snippet_budget = remaining_budget // max(1, len(remaining_files)) if remaining_files else 0
        
        # Add snippets for remaining files
        for f in remaining_files:
            path = f.get('path', 'unknown')
            content = f.get('content', '') or ''
            
            if not content:
                file_status[path] = 'empty'
                continue
            
            # Use the smaller of per-file budget and default max
            budget = min(per_file_snippet_budget, snippet_max_chars)
            snippet = select_relevant_snippet(content, path, cycle, budget)
            
            if snippet:
                snippets.append(f"--- FILE: {path} (SNIPPET) ---\n{snippet}")
                file_status[path] = 'snippet'
                used_chars += len(snippet)
    else:
        # Original behavior - just use snippets
        for f in files:
            path = f.get('path', 'unknown')
            content = f.get('content', '') or ''
            
            if not content:
                file_status[path] = 'empty'
                continue
            
            snippet = select_relevant_snippet(content, path, cycle, snippet_max_chars)
            if snippet:
                snippets.append(f"--- FILE: {path} ---\n{snippet}")
                file_status[path] = 'snippet'
    
    return "\n\n".join(snippets), file_status


def select_relevant_snippet(
    content: str,
    path: str,
    cycle: Dict[str, Any],
    max_chars: int = 4000,
) -> str:
    """Return a concise snippet of `content` focused on relevant regions.

    Heuristics (in order):
    - If content shorter than `max_chars`, return full content.
    - Find import lines that reference other cycle files/modules.
    - Find `def`/`class` definitions and include nearby context.
    - Find lines that mention node names from the cycle graph.
    - If nothing found, fall back to the file head truncated to `max_chars`.

    Args:
        content: Full file content.
        path: File path (used for language detection and token matching).
        cycle: Cycle spec dict with 'graph' containing 'nodes'.
        max_chars: Maximum characters for the returned snippet.

    Returns:
        Selected snippet string, possibly with truncation markers.
    """
    if not content:
        return ""

    if len(content) <= max_chars:
        return content

    lines = content.splitlines()
    n = len(lines)

    # Build a set of tokens to look for: graph node names and file basenames
    nodes = list(cycle.get("graph", {}).get("nodes", []))
    basename = Path(path).stem
    tokens: Set[str] = set(nodes + [basename])

    ranges: List[Tuple[int, int]] = []

    # Detect language and apply appropriate heuristics
    path_lower = path.lower()

    if path_lower.endswith('.cs'):
        ranges.extend(_extract_csharp_ranges(lines, tokens))
    elif path_lower.endswith('.py'):
        ranges.extend(_extract_python_ranges(lines, tokens))
    elif path_lower.endswith(('.js', '.ts', '.jsx', '.tsx')):
        ranges.extend(_extract_javascript_ranges(lines, tokens))
    elif path_lower.endswith(('.java', '.kt')):
        ranges.extend(_extract_java_ranges(lines, tokens))
    else:
        # Generic fallback
        ranges.extend(_extract_generic_ranges(lines, tokens))

    # Always add lines that mention any node/token
    ranges.extend(_extract_token_mentions(lines, tokens))

    if not ranges:
        # Fallback to head of file
        head = "\n".join(lines[: max(50, int(max_chars / 80))])
        if len(head) > max_chars:
            return head[:max_chars] + "\n...[truncated]"
        return head + "\n...[truncated]"

    # Merge overlapping ranges
    merged = _merge_ranges(ranges, n)

    # Build snippet from merged ranges
    parts = []
    total_len = 0
    for start, end in merged:
        part = "\n".join(lines[start:end])
        parts.append(part)
        total_len += len(part)
        if total_len > max_chars:
            break

    snippet = "\n\n...\n\n".join(parts)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "\n...[truncated]"
    return snippet


def _extract_python_ranges(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Extract relevant ranges for Python files."""
    ranges = []
    n = len(lines)
    import_re = re.compile(r"^\s*(from|import)\s+([\w\.]+)")
    def_re = re.compile(r"^\s*(def|class)\s+(\w+)")

    for i, line in enumerate(lines):
        # Import lines
        m = import_re.match(line)
        if m:
            target = m.group(2)
            for t in tokens:
                if t and (t in target or target.endswith(t)):
                    ranges.append((max(0, i - 2), min(n, i + 3)))
                    break

        # def/class definitions
        if def_re.match(line):
            ranges.append((max(0, i - 2), min(n, i + 10)))

    return ranges


def _extract_csharp_ranges(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Extract relevant ranges for C# files."""
    ranges = []
    n = len(lines)

    using_re = re.compile(r"^\s*(using)\s+([A-Za-z0-9_\.]+)\s*;")
    ns_re = re.compile(r"^\s*namespace\s+([A-Za-z0-9_\.]+)")
    type_re = re.compile(
        r"^\s*(public|private|internal|protected|static|sealed|partial|abstract|new|protected\s+internal)?\s*"
        r"(class|interface|struct|enum|record)\s+(\w+)"
    )
    method_re = re.compile(
        r"^\s*(public|private|internal|protected|static|async|virtual|override|sealed|extern)?\s*"
        r"[\w\<\>\[\],\s]+\s+\w+\s*\([^)]*\)\s*(\{|;|=>)"
    )

    for i, line in enumerate(lines):
        if using_re.match(line) or ns_re.match(line) or type_re.match(line) or method_re.match(line):
            ranges.append((max(0, i - 2), min(n, i + 6)))

    return ranges


def _extract_javascript_ranges(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Extract relevant ranges for JavaScript/TypeScript files."""
    ranges = []
    n = len(lines)

    import_re = re.compile(r"^\s*(import|export)\s+.*from\s+['\"]([^'\"]+)['\"]")
    require_re = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
    func_re = re.compile(r"^\s*(export\s+)?(async\s+)?(function|class|const|let|var)\s+(\w+)")

    for i, line in enumerate(lines):
        # Import/export statements
        if import_re.match(line) or require_re.search(line):
            ranges.append((max(0, i - 1), min(n, i + 2)))

        # Function/class definitions
        if func_re.match(line):
            ranges.append((max(0, i - 2), min(n, i + 10)))

    return ranges


def _extract_java_ranges(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Extract relevant ranges for Java/Kotlin files."""
    ranges = []
    n = len(lines)

    import_re = re.compile(r"^\s*import\s+([\w\.]+);?")
    class_re = re.compile(
        r"^\s*(public|private|protected)?\s*(static)?\s*(final)?\s*(abstract)?\s*"
        r"(class|interface|enum)\s+(\w+)"
    )
    method_re = re.compile(
        r"^\s*(public|private|protected)?\s*(static)?\s*(final)?\s*"
        r"[\w\<\>\[\],\s]+\s+\w+\s*\([^)]*\)\s*(\{|;|throws)"
    )

    for i, line in enumerate(lines):
        if import_re.match(line):
            ranges.append((max(0, i - 1), min(n, i + 2)))
        if class_re.match(line) or method_re.match(line):
            ranges.append((max(0, i - 2), min(n, i + 8)))

    return ranges


def _extract_generic_ranges(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Fallback extraction for unknown file types."""
    ranges = []
    n = len(lines)

    # Look for common patterns
    import_re = re.compile(r"^\s*(import|include|require|use)\s+")

    for i, line in enumerate(lines):
        if import_re.match(line):
            ranges.append((max(0, i - 1), min(n, i + 2)))

    return ranges


def _extract_token_mentions(lines: List[str], tokens: Set[str]) -> List[Tuple[int, int]]:
    """Find lines that mention any of the tokens."""
    ranges = []
    n = len(lines)

    for i, line in enumerate(lines):
        for t in tokens:
            if t and t in line:
                ranges.append((max(0, i - 2), min(n, i + 3)))
                break

    return ranges


def _merge_ranges(ranges: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent ranges."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged = [list(sorted_ranges[0])]

    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, min(e, n)) for s, e in merged]
