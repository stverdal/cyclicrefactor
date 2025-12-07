"""Patch application utilities for applying SEARCH/REPLACE operations.

This module provides functions to apply code patches to files:
- Atomic application (all-or-nothing)
- Partial rollback on failure
- Multiple matching strategies with confidence scoring
- Line hint disambiguation for duplicate patterns

Extracted from refactor_agent.py for better separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import re
import difflib

from utils.logging import get_logger
from utils.syntax_checker import (
    validate_code_block, check_truncation, get_common_indent, 
    normalize_line_endings
)

logger = get_logger("patch_applier")


@dataclass
class SearchReplaceResult:
    """Result of applying SEARCH/REPLACE operations."""
    content: str
    applied_count: int = 0
    total_count: int = 0
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    is_atomic: bool = True  # True if all-or-nothing was applied


@dataclass
class MatchResult:
    """Result of trying to find search text in content."""
    start: Optional[int]  # Start line index (0-based)
    end: Optional[int]    # End line index (exclusive)
    strategy: str         # Name of matching strategy used
    confidence: float     # Confidence score (0.0-1.0)


def extract_line_hint(text: str) -> Optional[int]:
    """Extract line number hint from text.
    
    Looks for patterns like:
    - // Line 42
    - // around line 100
    - # near line 50
    - (line 25)
    
    Args:
        text: Text to search for line hints
    
    Returns:
        Line number (0-indexed) if found, None otherwise
    """
    patterns = [
        r'(?://|#)\s*(?:line|near line|around line)\s*(\d+)',
        r'\(line\s*(\d+)\)',
        r'line\s*(\d+)\s*(?:$|:)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-indexed
    
    return None


def try_find_search_text(
    search_text: str, 
    content_lines: List[str],
    content_lines_stripped: List[str],
    line_hint: Optional[int] = None
) -> MatchResult:
    """Try to find search text in content using multiple strategies.
    
    Strategies (in order of preference):
    1. Exact line-by-line match
    2. Anchor-based matching with content similarity verification
    
    Args:
        search_text: The text to search for
        content_lines: Original file lines (with original whitespace)
        content_lines_stripped: Lines with trailing whitespace removed
        line_hint: Optional hint for which line the match should be near
        
    Returns:
        MatchResult with start/end lines, strategy used, and confidence
    """
    search_lines = [line.rstrip() for line in search_text.split('\n')]
    
    # Strategy 1: Exact line-by-line match (highest confidence)
    exact_matches = []
    for i in range(len(content_lines_stripped) - len(search_lines) + 1):
        if content_lines_stripped[i:i + len(search_lines)] == search_lines:
            exact_matches.append((i, i + len(search_lines)))
    
    if exact_matches:
        if len(exact_matches) == 1:
            return MatchResult(exact_matches[0][0], exact_matches[0][1], "exact", 1.0)
        else:
            # Multiple exact matches - use line hint to disambiguate
            logger.warning(f"Found {len(exact_matches)} exact matches for search text, using line hint: {line_hint}")
            if line_hint is not None:
                # Pick the match closest to the hint
                best_match = min(exact_matches, key=lambda m: abs(m[0] - line_hint))
                return MatchResult(best_match[0], best_match[1], "exact_with_hint", 0.95)
            else:
                # No hint - use first match but lower confidence
                logger.warning("No line hint provided for duplicate pattern - using first match")
                return MatchResult(exact_matches[0][0], exact_matches[0][1], "exact_first", 0.7)
    
    # Strategy 2: Anchor-based matching with content similarity verification
    non_empty_search = [l.strip() for l in search_lines if l.strip()]
    if len(non_empty_search) >= 2:
        first_anchor = non_empty_search[0]
        last_anchor = non_empty_search[-1]
        
        # Find ALL potential matches, not just the first
        candidates = []
        i = 0
        while i < len(content_lines):
            stripped = content_lines[i].strip()
            if stripped == first_anchor:
                # Found a potential start, look for matching end
                for j in range(i + 1, len(content_lines)):
                    if content_lines[j].strip() == last_anchor:
                        candidates.append((i, j + 1))  # exclusive end
                        break
            i += 1
        
        # Evaluate each candidate by content similarity
        scored_matches = []
        
        for start_idx, end_idx in candidates:
            # Check size reasonableness first
            expected_size = len(search_lines)
            actual_size = end_idx - start_idx
            size_ratio = actual_size / expected_size if expected_size > 0 else 0
            
            if not (0.5 <= size_ratio <= 2.0):
                continue  # Size too different, skip
            
            # Calculate content similarity using difflib
            matched_content = "\n".join(content_lines[start_idx:end_idx])
            similarity = difflib.SequenceMatcher(
                None, 
                search_text.strip(), 
                matched_content.strip()
            ).ratio()
            
            # Weight by size match as well
            size_confidence = 1.0 - (abs(1.0 - size_ratio) * 0.3)
            combined_score = (similarity * 0.7) + (size_confidence * 0.3)
            
            # Add line_hint proximity bonus (up to 0.1)
            hint_bonus = 0.0
            if line_hint is not None:
                distance = abs(start_idx - line_hint)
                # Closer to hint = higher bonus
                hint_bonus = max(0, 0.1 - (distance / 100 * 0.1))
            
            final_score = combined_score + hint_bonus
            scored_matches.append((start_idx, end_idx, final_score, combined_score))
        
        if scored_matches:
            # Sort by final score (descending)
            scored_matches.sort(key=lambda x: x[2], reverse=True)
            best = scored_matches[0]
            start_idx, end_idx, final_score, base_score = best
            
            if base_score >= 0.5:
                # Log if line hint was used
                if line_hint is not None and len(scored_matches) > 1:
                    logger.debug(f"Line hint {line_hint} used to disambiguate {len(scored_matches)} anchor matches")
                
                # Scale confidence: 0.5 similarity -> 0.4 confidence, 1.0 -> 0.8
                confidence = 0.4 + (base_score - 0.5) * 0.8
                logger.debug(f"Anchor match: lines {start_idx}-{end_idx}, similarity={base_score:.2f}, confidence={confidence:.2f}")
                return MatchResult(start_idx, end_idx, "anchor", min(0.8, confidence))
    
    # Strategy 3: Fuzzy line-by-line matching (for LLM-hallucinated whitespace/minor differences)
    # This is useful when the LLM generates code that's "close" but not exact
    best_fuzzy_match = None
    best_fuzzy_score = 0.0
    
    for i in range(len(content_lines_stripped) - len(search_lines) + 1):
        candidate_lines = content_lines_stripped[i:i + len(search_lines)]
        
        # Calculate line-by-line similarity
        line_scores = []
        for search_line, content_line in zip(search_lines, candidate_lines):
            # Compare stripped versions (ignore leading/trailing whitespace)
            s_stripped = search_line.strip()
            c_stripped = content_line.strip()
            
            if s_stripped == c_stripped:
                line_scores.append(1.0)
            elif not s_stripped and not c_stripped:
                line_scores.append(1.0)  # Both empty
            else:
                # Use sequence matcher for partial matches
                ratio = difflib.SequenceMatcher(None, s_stripped, c_stripped).ratio()
                line_scores.append(ratio)
        
        if line_scores:
            avg_score = sum(line_scores) / len(line_scores)
            # Penalize matches where individual lines are very different
            min_line_score = min(line_scores) if line_scores else 0
            adjusted_score = avg_score * 0.7 + min_line_score * 0.3
            
            # Apply line hint bonus
            if line_hint is not None:
                distance = abs(i - line_hint)
                hint_bonus = max(0, 0.05 - (distance / 200 * 0.05))
                adjusted_score += hint_bonus
            
            if adjusted_score > best_fuzzy_score:
                best_fuzzy_score = adjusted_score
                best_fuzzy_match = (i, i + len(search_lines), adjusted_score, avg_score, min_line_score)
    
    if best_fuzzy_match and best_fuzzy_score >= 0.75:
        start_idx, end_idx, score, avg, min_score = best_fuzzy_match
        
        # Special case: near-perfect similarity should get higher confidence
        # This handles LLM-generated search text that's correct but has minor whitespace differences
        if avg >= 0.95 and min_score >= 0.90:
            # High similarity = trust it more (0.95 avg -> 0.75 confidence, 1.0 -> 0.85)
            confidence = 0.75 + (avg - 0.95) * 2.0
            confidence = min(0.85, confidence)
            logger.debug(f"Fuzzy match (high-similarity): lines {start_idx}-{end_idx}, avg_sim={avg:.2f}, min_line={min_score:.2f}, confidence={confidence:.2f}")
            return MatchResult(start_idx, end_idx, "fuzzy_high_sim", confidence)
        
        # Standard fuzzy match: lower confidence (max 0.65)
        confidence = 0.4 + (score - 0.75) * 1.0  # 0.75->0.4, 1.0->0.65
        confidence = min(0.65, max(0.4, confidence))
        logger.debug(f"Fuzzy match: lines {start_idx}-{end_idx}, avg_sim={avg:.2f}, min_line={min_score:.2f}, confidence={confidence:.2f}")
        return MatchResult(start_idx, end_idx, "fuzzy", confidence)
    
    # No match found - log diagnostic information
    if search_lines:
        first_line = search_lines[0].strip()[:60]
        last_line = search_lines[-1].strip()[:60] if len(search_lines) > 1 else ""
        logger.debug(f"No match found for search text ({len(search_lines)} lines)")
        logger.debug(f"  First line: '{first_line}'")
        if last_line:
            logger.debug(f"  Last line: '{last_line}'")
        
        # Try to find partial matches for diagnostics
        if first_line:
            partial_matches = []
            for i, line in enumerate(content_lines_stripped):
                if first_line in line.strip() or difflib.SequenceMatcher(None, first_line, line.strip()).ratio() > 0.8:
                    partial_matches.append((i, line.strip()[:60]))
            if partial_matches:
                logger.debug(f"  Possible partial matches for first line:")
                for line_num, text in partial_matches[:3]:
                    logger.debug(f"    Line {line_num + 1}: '{text}'")
    
    return MatchResult(None, None, "", 0.0)


def apply_search_replace_atomic(
    original: str, 
    search_replace_blocks: str,
    min_confidence: float = 0.7,
    warn_confidence: float = 0.85,
    allow_low_confidence: bool = False
) -> SearchReplaceResult:
    """Apply SEARCH/REPLACE blocks atomically (all or nothing).
    
    Format:
    <<<<<<< SEARCH
    old text
    =======
    new text
    >>>>>>> REPLACE
    
    Args:
        original: Original file content
        search_replace_blocks: The SEARCH/REPLACE block content
        min_confidence: Minimum confidence required (0.0-1.0)
        warn_confidence: Warn if confidence below this (but still apply)
        allow_low_confidence: If True, accept matches below min_confidence with warning
        
    Returns:
        SearchReplaceResult with applied content or original if failed
    """
    warnings = []
    original = normalize_line_endings(original)
    search_replace_blocks = normalize_line_endings(search_replace_blocks)
    
    # Parse SEARCH/REPLACE blocks
    pattern = r'<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE'
    matches = re.findall(pattern, search_replace_blocks)
    
    if not matches:
        logger.warning("No SEARCH/REPLACE blocks found in content")
        return SearchReplaceResult(
            content=original,
            applied_count=0,
            total_count=0,
            warnings=["No SEARCH/REPLACE blocks parsed from LLM output"],
            confidence=0.0,
            is_atomic=True
        )
    
    logger.info(f"Parsed {len(matches)} SEARCH/REPLACE blocks, attempting atomic apply")
    
    # Phase 1: Dry run - verify all blocks can be found
    content_lines = original.split('\n')
    content_lines_stripped = [line.rstrip() for line in content_lines]
    
    block_plans = []
    all_found = True
    total_confidence = 0.0
    
    for idx, (search_text, replace_text) in enumerate(matches):
        block_num = idx + 1
        
        # Check for truncated replace text
        truncation = check_truncation(replace_text, f"block_{block_num}")
        if truncation:
            warning = f"Block {block_num}: Replace text appears truncated - {truncation.message}"
            logger.warning(warning)
            warnings.append(warning)
            all_found = False
            continue
        
        # Try to extract line hint from search text (for duplicate pattern disambiguation)
        line_hint = extract_line_hint(search_text)
        
        # Try to find the search text
        match_result = try_find_search_text(
            search_text, content_lines, content_lines_stripped, line_hint
        )
        
        if match_result.start is None:
            warning = f"Block {block_num}: Search text not found (first 80 chars: {search_text[:80]}...)"
            logger.warning(warning)
            warnings.append(warning)
            all_found = False
        elif match_result.confidence < min_confidence:
            warning = f"Block {block_num}: Low confidence match ({match_result.confidence:.2f} < {min_confidence}) using {match_result.strategy}"
            logger.warning(warning)
            warnings.append(warning)
            if allow_low_confidence:
                # Accept the match anyway with warning
                logger.warning(f"Block {block_num}: Accepting low confidence match due to allow_low_confidence=True")
                block_plans.append({
                    "block_num": block_num,
                    "start": match_result.start,
                    "end": match_result.end,
                    "replace_text": replace_text,
                    "confidence": match_result.confidence,
                    "strategy": match_result.strategy,
                    "original_lines": content_lines[match_result.start:match_result.end],
                    "low_confidence_warning": True,
                })
                total_confidence += match_result.confidence
            else:
                all_found = False
        else:
            # Check for warn_confidence threshold
            if match_result.confidence < warn_confidence:
                warning = f"Block {block_num}: Marginal confidence ({match_result.confidence:.2f} < {warn_confidence}), proceeding with caution"
                logger.warning(warning)
                warnings.append(warning)
            block_plans.append({
                "block_num": block_num,
                "start": match_result.start,
                "end": match_result.end,
                "replace_text": replace_text,
                "confidence": match_result.confidence,
                "strategy": match_result.strategy,
                "original_lines": content_lines[match_result.start:match_result.end],
            })
            total_confidence += match_result.confidence
            logger.debug(f"Block {block_num}: Found with {match_result.strategy} strategy (confidence: {match_result.confidence:.2f})")
    
    # Phase 2: Check for overlapping blocks
    block_plans.sort(key=lambda x: x["start"])
    for i in range(len(block_plans) - 1):
        if block_plans[i]["end"] > block_plans[i + 1]["start"]:
            warning = f"Blocks {block_plans[i]['block_num']} and {block_plans[i+1]['block_num']} overlap"
            logger.warning(warning)
            warnings.append(warning)
            all_found = False
    
    # If not all blocks found/valid, return original (atomic failure)
    if not all_found:
        logger.warning(f"Atomic apply failed: {len(block_plans)}/{len(matches)} blocks valid, returning original")
        return SearchReplaceResult(
            content=original,
            applied_count=0,
            total_count=len(matches),
            warnings=warnings,
            confidence=0.0,
            is_atomic=True
        )
    
    # Phase 3: Apply all blocks (in reverse order to preserve line numbers)
    result_lines = content_lines.copy()
    block_plans.sort(key=lambda x: x["start"], reverse=True)
    
    for plan in block_plans:
        # Preserve indentation
        original_indent = get_common_indent(plan["original_lines"])
        replace_lines = plan["replace_text"].split('\n')
        replace_indent = get_common_indent(replace_lines)
        
        if not replace_indent and original_indent:
            replace_lines = [original_indent + line if line.strip() else line for line in replace_lines]
        
        result_lines = result_lines[:plan["start"]] + replace_lines + result_lines[plan["end"]:]
        logger.debug(f"Block {plan['block_num']}: Applied at lines {plan['start']}-{plan['end']}")
    
    result = '\n'.join(result_lines)
    avg_confidence = total_confidence / len(matches) if matches else 0.0
    
    logger.info(f"Atomic apply succeeded: {len(matches)}/{len(matches)} blocks applied (avg confidence: {avg_confidence:.2f})")
    
    return SearchReplaceResult(
        content=result,
        applied_count=len(matches),
        total_count=len(matches),
        warnings=warnings,
        confidence=avg_confidence,
        is_atomic=True
    )


def apply_search_replace_list_atomic(
    original: str, 
    search_replace_list: List[Dict[str, str]],
    min_confidence: float = 0.7,
    warn_confidence: float = 0.85,
    allow_low_confidence: bool = False
) -> SearchReplaceResult:
    """Apply a list of search/replace operations atomically.
    
    Args:
        original: Original file content
        search_replace_list: List of {search, replace} dicts
        min_confidence: Minimum confidence required
        warn_confidence: Warn if confidence below this
        allow_low_confidence: If True, accept matches below min_confidence
        
    Returns:
        SearchReplaceResult with applied content or original if failed
    """
    warnings = []
    original = normalize_line_endings(original)
    
    if not search_replace_list:
        return SearchReplaceResult(
            content=original,
            applied_count=0,
            total_count=0,
            warnings=["Empty search/replace list"],
            confidence=0.0,
            is_atomic=True
        )
    
    logger.info(f"Applying {len(search_replace_list)} search/replace operations atomically")
    
    # Phase 1: Dry run
    content_lines = original.split('\n')
    content_lines_stripped = [line.rstrip() for line in content_lines]
    
    block_plans = []
    all_found = True
    total_confidence = 0.0
    
    for idx, sr in enumerate(search_replace_list):
        op_num = idx + 1
        search_text = normalize_line_endings(sr.get("search", ""))
        replace_text = normalize_line_endings(sr.get("replace", ""))
        
        if not search_text:
            warnings.append(f"Op {op_num}: Empty search text")
            all_found = False
            continue
        
        # Check for truncation
        truncation = check_truncation(replace_text, f"op_{op_num}")
        if truncation:
            warning = f"Op {op_num}: Replace text appears truncated"
            logger.warning(warning)
            warnings.append(warning)
            all_found = False
            continue
        
        # Try to extract line hint from search text
        line_hint = extract_line_hint(search_text)
        
        # Also check for line_start hint in the dict
        if line_hint is None and isinstance(sr, dict):
            hint_from_dict = sr.get("line") or sr.get("line_start")
            if hint_from_dict is not None:
                line_hint = int(hint_from_dict) - 1  # Convert to 0-indexed
        
        # Try to find
        match_result = try_find_search_text(
            search_text, content_lines, content_lines_stripped, line_hint
        )
        
        if match_result.start is None:
            # Log more diagnostic info about the failed match
            search_preview = search_text[:100].replace('\n', '\\n')
            warning = f"Op {op_num}: Search text not found"
            logger.warning(warning)
            logger.debug(f"Op {op_num}: Search preview: '{search_preview}...'")
            logger.debug(f"Op {op_num}: File has {len(content_lines)} lines")
            warnings.append(warning)
            all_found = False
        elif match_result.confidence < min_confidence:
            warning = f"Op {op_num}: Low confidence match ({match_result.confidence:.2f} < {min_confidence})"
            logger.warning(warning)
            warnings.append(warning)
            if allow_low_confidence:
                logger.warning(f"Op {op_num}: Accepting low confidence match due to allow_low_confidence=True")
                block_plans.append({
                    "op_num": op_num,
                    "start": match_result.start,
                    "end": match_result.end,
                    "replace_text": replace_text,
                    "confidence": match_result.confidence,
                    "strategy": match_result.strategy,
                    "original_lines": content_lines[match_result.start:match_result.end],
                    "low_confidence_warning": True,
                })
                total_confidence += match_result.confidence
            else:
                all_found = False
        else:
            if match_result.confidence < warn_confidence:
                warning = f"Op {op_num}: Marginal confidence ({match_result.confidence:.2f} < {warn_confidence}), proceeding with caution"
                logger.warning(warning)
                warnings.append(warning)
            block_plans.append({
                "op_num": op_num,
                "start": match_result.start,
                "end": match_result.end,
                "replace_text": replace_text,
                "confidence": match_result.confidence,
                "strategy": match_result.strategy,
                "original_lines": content_lines[match_result.start:match_result.end],
            })
            total_confidence += match_result.confidence
    
    # Check overlaps
    block_plans.sort(key=lambda x: x["start"])
    for i in range(len(block_plans) - 1):
        if block_plans[i]["end"] > block_plans[i + 1]["start"]:
            warning = f"Ops {block_plans[i]['op_num']} and {block_plans[i+1]['op_num']} overlap"
            logger.warning(warning)
            warnings.append(warning)
            all_found = False
    
    if not all_found:
        logger.warning(f"Atomic apply failed: {len(block_plans)}/{len(search_replace_list)} ops valid")
        # Log which operations failed
        failed_ops = [i + 1 for i in range(len(search_replace_list)) if not any(p.get("op_num") == i + 1 for p in block_plans)]
        if failed_ops:
            logger.debug(f"Failed operations: {failed_ops}")
        return SearchReplaceResult(
            content=original,
            applied_count=0,
            total_count=len(search_replace_list),
            warnings=warnings,
            confidence=0.0,
            is_atomic=True
        )
    
    # Phase 2: Apply in reverse order
    result_lines = content_lines.copy()
    block_plans.sort(key=lambda x: x["start"], reverse=True)
    
    for plan in block_plans:
        original_indent = get_common_indent(plan["original_lines"])
        replace_lines = plan["replace_text"].split('\n')
        replace_indent = get_common_indent(replace_lines)
        
        if not replace_indent and original_indent:
            replace_lines = [original_indent + line if line.strip() else line for line in replace_lines]
        
        result_lines = result_lines[:plan["start"]] + replace_lines + result_lines[plan["end"]:]
    
    result = '\n'.join(result_lines)
    avg_confidence = total_confidence / len(search_replace_list) if search_replace_list else 0.0
    
    logger.info(f"Atomic apply succeeded: {len(search_replace_list)} ops (avg confidence: {avg_confidence:.2f})")
    
    return SearchReplaceResult(
        content=result,
        applied_count=len(search_replace_list),
        total_count=len(search_replace_list),
        warnings=warnings,
        confidence=avg_confidence,
        is_atomic=True
    )


def apply_with_partial_rollback(
    original: str,
    search_replace_blocks: str,
    path: str,
    original_content: str = None,
    min_confidence: float = 0.7,
    warn_confidence: float = 0.85,
    allow_low_confidence: bool = False
) -> SearchReplaceResult:
    """Apply SEARCH/REPLACE blocks with partial rollback on validation failure.
    
    If the full atomic apply succeeds but validation fails, this method
    will try to identify and skip the problematic block(s).
    
    Args:
        original: Original file content
        search_replace_blocks: The SEARCH/REPLACE block content
        path: File path for validation context
        original_content: Original content for introduced-issue checking
        min_confidence: Minimum confidence required
        warn_confidence: Warn if confidence below this
        allow_low_confidence: If True, accept matches below min_confidence
        
    Returns:
        SearchReplaceResult with best valid content found
    """
    # First, try full atomic apply
    full_result = apply_search_replace_atomic(
        original, search_replace_blocks, 
        min_confidence, warn_confidence, allow_low_confidence
    )
    
    if full_result.applied_count == 0:
        # Atomic apply failed entirely, nothing to partially rollback
        return full_result
    
    # Validate the result
    validation = validate_code_block(full_result.content, path, original_content)
    
    if validation.is_valid:
        # Full apply is valid, return it
        return full_result
    
    # Full apply has issues - try to identify problematic blocks
    logger.warning(f"Full atomic apply has validation issues, attempting partial rollback")
    
    # Parse blocks again
    pattern = r'<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE'
    matches = list(re.findall(pattern, normalize_line_endings(search_replace_blocks)))
    
    if len(matches) <= 1:
        # Only one block, can't do partial rollback
        logger.warning("Only one block, cannot do partial rollback")
        full_result.warnings.append("Partial rollback not possible with single block")
        return full_result
    
    # Try removing blocks one at a time to find the problematic one
    best_result = None
    best_applied = 0
    
    for skip_idx in range(len(matches)):
        # Build new block string without the skipped block
        new_blocks = []
        for i, (search, replace) in enumerate(matches):
            if i != skip_idx:
                new_blocks.append(f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
        
        if not new_blocks:
            continue
        
        partial_blocks = "\n\n".join(new_blocks)
        partial_result = apply_search_replace_atomic(
            original, partial_blocks,
            min_confidence, warn_confidence, allow_low_confidence
        )
        
        if partial_result.applied_count == 0:
            continue
        
        # Validate this partial result
        partial_validation = validate_code_block(partial_result.content, path, original_content)
        
        if partial_validation.is_valid and partial_result.applied_count > best_applied:
            best_result = partial_result
            best_applied = partial_result.applied_count
            best_result.warnings.append(f"Block {skip_idx + 1} skipped due to validation issues")
            logger.info(f"Found valid partial apply by skipping block {skip_idx + 1}: {best_applied}/{len(matches)} blocks")
    
    if best_result:
        best_result.is_atomic = False  # Mark as partial
        return best_result
    
    # No valid partial found - try combinations (skip 2 blocks)
    if len(matches) >= 3:
        for skip_i in range(len(matches)):
            for skip_j in range(skip_i + 1, len(matches)):
                new_blocks = []
                for i, (search, replace) in enumerate(matches):
                    if i not in (skip_i, skip_j):
                        new_blocks.append(f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
                
                if not new_blocks:
                    continue
                
                partial_blocks = "\n\n".join(new_blocks)
                partial_result = apply_search_replace_atomic(
                    original, partial_blocks,
                    min_confidence, warn_confidence, allow_low_confidence
                )
                
                if partial_result.applied_count == 0:
                    continue
                
                partial_validation = validate_code_block(partial_result.content, path, original_content)
                
                if partial_validation.is_valid and partial_result.applied_count > best_applied:
                    best_result = partial_result
                    best_applied = partial_result.applied_count
                    best_result.warnings.append(f"Blocks {skip_i + 1} and {skip_j + 1} skipped due to validation issues")
                    logger.info(f"Found valid partial by skipping blocks {skip_i + 1},{skip_j + 1}: {best_applied}/{len(matches)}")
    
    if best_result:
        best_result.is_atomic = False
        return best_result
    
    # Could not find valid partial - return the full result with warning
    logger.warning("Partial rollback unsuccessful, returning full (invalid) result")
    full_result.warnings.append("Partial rollback attempted but no valid subset found")
    return full_result


def apply_search_replace_simple(original: str, search_replace_blocks: str) -> Tuple[str, List[str]]:
    """Apply SEARCH/REPLACE blocks to original content (simple non-atomic version).
    
    This is the legacy interface that returns a tuple instead of SearchReplaceResult.
    
    Format:
    <<<<<<< SEARCH
    old text
    =======
    new text
    >>>>>>> REPLACE
    
    Args:
        original: Original file content
        search_replace_blocks: The SEARCH/REPLACE block content
    
    Returns:
        Tuple of (result_content, list_of_warnings)
    """
    result = apply_search_replace_atomic(original, search_replace_blocks)
    return result.content, result.warnings


def apply_search_replace_list_simple(
    original: str, 
    search_replace_list: List[Dict[str, str]]
) -> Tuple[str, List[str]]:
    """Apply a list of search/replace operations (simple non-atomic version).
    
    This is the legacy interface that returns a tuple instead of SearchReplaceResult.
    
    Args:
        original: Original file content
        search_replace_list: List of {search, replace} dicts
    
    Returns:
        Tuple of (result_content, list_of_warnings)
    """
    result = apply_search_replace_list_atomic(original, search_replace_list)
    return result.content, result.warnings


def apply_append_prepend(
    original: str,
    append: Optional[str] = None,
    prepend: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Apply append/prepend content to a file.
    
    This handles adding new content to files without needing SEARCH/REPLACE.
    Use append to add content at the end of the file (e.g., new interfaces, classes).
    Use prepend to add content at the start of the file (e.g., new imports).
    
    Args:
        original: Original file content
        append: Optional content to add at end of file
        prepend: Optional content to add at start of file
    
    Returns:
        Tuple of (result_content, list_of_warnings)
    """
    original = normalize_line_endings(original)
    result = original
    warnings = []
    
    # Apply prepend first
    if prepend:
        prepend = normalize_line_endings(prepend)
        # Check if it looks truncated
        truncation = check_truncation(prepend, "prepend")
        if truncation:
            warnings.append(f"Prepend content may be truncated: {truncation}")
        
        # Add prepend content, ensure proper line separation
        if result and not prepend.endswith('\n'):
            prepend = prepend + '\n'
        result = prepend + result
        logger.info(f"Prepended {len(prepend)} chars to file")
    
    # Apply append
    if append:
        append = normalize_line_endings(append)
        # Check if it looks truncated
        truncation = check_truncation(append, "append")
        if truncation:
            warnings.append(f"Append content may be truncated: {truncation}")
        
        # Add append content, ensure proper line separation
        if result and not result.endswith('\n'):
            result = result + '\n'
        # Ensure there's a blank line before new content if it's a class/interface
        if append.strip().startswith(('public ', 'internal ', 'private ', 'namespace ', 'using ', 'class ', 'interface ')):
            if result and not result.endswith('\n\n'):
                result = result.rstrip('\n') + '\n\n'
        result = result + append
        logger.info(f"Appended {len(append)} chars to file")
    
    return result, warnings
