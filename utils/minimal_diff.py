"""Minimal diff mode for breaking cycles with smallest possible change.

This module implements a focused approach that:
1. Identifies the single weakest edge in the cycle
2. Generates only the minimal change needed to break that edge
3. Prioritizes simple changes (parameter injection, lazy import) over complex ones

Benefits:
- Much higher success rate than full refactoring
- Produces clear, reviewable diffs
- Good for demonstrations and incremental progress
- Less prone to hallucination (smaller scope)
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logging import get_logger
from utils.prompt_loader import load_template, safe_format

logger = get_logger("minimal_diff")


@dataclass
class MinimalDiffResult:
    """Result of minimal diff analysis and generation."""
    success: bool = False
    target_edge: Optional[Dict[str, str]] = None  # from, to, import_statement
    strategy: str = ""  # parameter_injection, lazy_import, callback, interface_extraction
    rationale: str = ""
    patch: Optional[Dict[str, Any]] = None  # path, search_replace, changes_made
    new_file: Optional[Dict[str, Any]] = None  # path, content (if needed)
    remaining_cycle: str = ""  # Description of what remains
    raw_llm_response: str = ""
    confidence: float = 0.0


def get_weakest_edge_info(cycle_spec: Dict[str, Any]) -> str:
    """Extract weakest edge information from cycle spec.
    
    Args:
        cycle_spec: Cycle specification dict
        
    Returns:
        Formatted string describing the weakest edge
    """
    weakest = cycle_spec.get("weakest_edge")
    
    if not weakest:
        # Fallback: pick first edge from graph
        edges = cycle_spec.get("graph", {}).get("edges", [])
        if edges:
            return f"First edge in cycle: {edges[0][0]} → {edges[0][1]}\n(No weakest edge analysis available)"
        return "No edge information available"
    
    source = weakest.get("source", "unknown")
    target = weakest.get("target", "unknown")
    weight = weakest.get("weight", "unknown")
    reason = weakest.get("reason", "lowest coupling")
    
    return f"""**Recommended edge to break**: `{source}` → `{target}`
- Coupling strength: {weight} references
- Reason: {reason}

Focus your change on removing the import of `{target}` from `{source}`.
"""


def build_minimal_diff_prompt(
    cycle_spec: Dict[str, Any],
    file_snippets: str,
    prompts_dir: str = "prompts",
) -> str:
    """Build the prompt for minimal diff mode.
    
    Args:
        cycle_spec: Cycle specification
        file_snippets: Formatted file content
        prompts_dir: Directory containing prompt templates
        
    Returns:
        Formatted prompt string
    """
    template_path = Path(prompts_dir) / "prompt_minimal_diff.txt"
    
    weakest_edge_info = get_weakest_edge_info(cycle_spec)
    
    if template_path.exists():
        tpl = load_template(str(template_path))
        return safe_format(
            tpl,
            id=cycle_spec.get("id", "unknown"),
            graph=json.dumps(cycle_spec.get("graph", {})),
            weakest_edge_info=weakest_edge_info,
            file_snippets=file_snippets,
        )
    
    # Fallback inline template
    return f"""Generate the MINIMAL change to break this cycle.

## Cycle: {cycle_spec.get('id', 'unknown')}
Graph: {json.dumps(cycle_spec.get('graph', {}))}

## Weakest Edge
{weakest_edge_info}

## Files
{file_snippets}

Output JSON with: target_edge, strategy, patch (path, search_replace), rationale.
Focus on ONE small change - prefer parameter injection or lazy import over creating new files.
"""


def parse_minimal_diff_response(response: str) -> MinimalDiffResult:
    """Parse LLM response for minimal diff.
    
    Args:
        response: LLM response string
        
    Returns:
        MinimalDiffResult with parsed data
    """
    result = MinimalDiffResult(raw_llm_response=response)
    
    # Try to extract JSON
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        logger.error("No JSON found in minimal diff response")
        return result
    
    try:
        data = json.loads(json_match.group())
        
        result.target_edge = data.get("target_edge")
        result.strategy = data.get("strategy", "")
        result.rationale = data.get("rationale", "")
        result.patch = data.get("patch")
        result.remaining_cycle = data.get("remaining_cycle", "")
        
        # Handle new_file
        new_file_data = data.get("new_file", {})
        if new_file_data and new_file_data.get("needed"):
            result.new_file = new_file_data
        
        # Calculate confidence based on strategy
        strategy_confidence = {
            "parameter_injection": 0.9,  # Most reliable
            "lazy_import": 0.85,
            "callback": 0.8,
            "interface_extraction": 0.6,  # Most likely to have issues
        }
        result.confidence = strategy_confidence.get(result.strategy, 0.5)
        
        # Validate that we have required fields
        if result.patch and result.target_edge:
            result.success = True
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse minimal diff JSON: {e}")
    
    return result


def validate_minimal_diff(result: MinimalDiffResult, files: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate the minimal diff result.
    
    Args:
        result: Parsed MinimalDiffResult
        files: Original file list for validation
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    if not result.patch:
        issues.append("No patch generated")
        return False, issues
    
    patch_path = result.patch.get("path", "")
    if not patch_path:
        issues.append("Patch has no file path")
    
    # Check that patch file exists in cycle
    file_paths = [f.get("path", "") for f in files]
    path_exists = any(
        patch_path == fp or patch_path.endswith(fp.split('/')[-1].split('\\')[-1])
        for fp in file_paths
    )
    if not path_exists:
        issues.append(f"Patch targets file not in cycle: {patch_path}")
    
    # Check for search_replace content
    sr_list = result.patch.get("search_replace", [])
    if not sr_list:
        issues.append("No search/replace blocks in patch")
    
    for i, sr in enumerate(sr_list):
        search = sr.get("search", "")
        replace = sr.get("replace", "")
        
        if not search:
            issues.append(f"Search block {i} is empty")
        if search == replace:
            issues.append(f"No-op patch detected: search == replace in block {i}")
    
    # Validate that target edge is being addressed
    if result.target_edge:
        import_stmt = result.target_edge.get("import_statement", "")
        if import_stmt:
            # Check if the import is being removed or modified
            found_import_change = False
            for sr in sr_list:
                search = sr.get("search", "")
                replace = sr.get("replace", "")
                # Check if import appears in search but not in replace (being removed)
                if import_stmt in search or any(part in search for part in import_stmt.split()):
                    found_import_change = True
                    break
            
            if not found_import_change:
                issues.append(f"Patch doesn't appear to modify the target import: {import_stmt[:50]}")
    
    return len(issues) == 0, issues


def run_minimal_diff_mode(
    cycle_spec: Dict[str, Any],
    files: List[Dict[str, Any]],
    file_snippets: str,
    llm_call_fn,
    prompts_dir: str = "prompts",
) -> MinimalDiffResult:
    """Run minimal diff mode to generate a focused, small change.
    
    Args:
        cycle_spec: Cycle specification dict
        files: List of file dicts with path and content
        file_snippets: Pre-formatted file snippets
        llm_call_fn: Function to call LLM
        prompts_dir: Directory containing prompts
        
    Returns:
        MinimalDiffResult with the generated patch
    """
    logger.info("Running minimal diff mode")
    
    # Build prompt
    prompt = build_minimal_diff_prompt(cycle_spec, file_snippets, prompts_dir)
    
    try:
        # Call LLM
        response = llm_call_fn(prompt)
        
        # Parse response
        result = parse_minimal_diff_response(str(response))
        
        if result.success:
            logger.info(f"Minimal diff generated: strategy={result.strategy}, "
                       f"target={result.target_edge}")
            
            # Validate
            is_valid, issues = validate_minimal_diff(result, files)
            if not is_valid:
                logger.warning(f"Minimal diff validation issues: {issues}")
                # Don't fail, but reduce confidence
                result.confidence *= 0.7
        else:
            logger.warning("Failed to generate minimal diff")
        
        return result
        
    except Exception as e:
        logger.error(f"Minimal diff mode failed: {e}")
        return MinimalDiffResult(success=False, raw_llm_response=str(e))


def format_minimal_diff_summary(result: MinimalDiffResult) -> str:
    """Format minimal diff result as human-readable summary.
    
    Args:
        result: MinimalDiffResult
        
    Returns:
        Formatted summary string
    """
    lines = ["## Minimal Diff Analysis", ""]
    
    if result.target_edge:
        lines.append(f"**Target Edge**: `{result.target_edge.get('from')}` → `{result.target_edge.get('to')}`")
        if result.target_edge.get('import_statement'):
            lines.append(f"**Import to Remove**: `{result.target_edge.get('import_statement')}`")
        lines.append("")
    
    lines.append(f"**Strategy**: {result.strategy}")
    lines.append(f"**Confidence**: {result.confidence:.0%}")
    lines.append("")
    
    if result.rationale:
        lines.append(f"**Rationale**: {result.rationale}")
        lines.append("")
    
    if result.patch:
        lines.append(f"**File to Modify**: `{result.patch.get('path')}`")
        changes = result.patch.get("changes_made", [])
        if changes:
            lines.append("**Changes**:")
            for c in changes:
                lines.append(f"  - {c}")
        lines.append("")
    
    if result.new_file:
        lines.append(f"**New File Needed**: `{result.new_file.get('path')}`")
        lines.append("")
    
    if result.remaining_cycle:
        lines.append(f"**After This Change**: {result.remaining_cycle}")
    
    return "\n".join(lines)
