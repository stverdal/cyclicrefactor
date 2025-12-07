"""Utils package - Utility modules for the cycle-refactoring pipeline.

This package provides shared utilities used across agents and orchestration:
- patch_parser: Parse LLM outputs into structured patches
- patch_applier: Apply SEARCH/REPLACE operations with confidence scoring
- diff_utils: Generate diffs and validate content changes
- syntax_checker: Validate code syntax and detect truncation
- compile_checker: Run compilation/lint checks on files
- context_budget: Manage token budgets for LLM context
- input_normalizer: Normalize cycle input data
- logging: Centralized logging configuration
- persistence: Save/load state to disk
- prompt_loader: Load prompt templates
- rag_query_builder: Build RAG queries for context retrieval
- snippet_selector: Select relevant code snippets
- cycle_verifier: Verify cycle resolution
"""

# Patch parsing utilities
from utils.patch_parser import (
    parse_json_patches,
    parse_marker_patches,
    parse_search_replace_json,
    infer_patches,
    clean_code_content,
    extract_patches_from_data,
    extract_patches_lenient,
    extract_json_string_value,
)

# Patch application utilities
from utils.patch_applier import (
    SearchReplaceResult,
    MatchResult,
    apply_search_replace_atomic,
    apply_search_replace_list_atomic,
    apply_with_partial_rollback,
    try_find_search_text,
    extract_line_hint,
)

# Diff and validation utilities
from utils.diff_utils import (
    make_unified_diff,
    looks_truncated,
    get_common_indent,
    validate_patched_content,
    check_for_truncation,
    compute_diff_stats,
)

# Syntax checking utilities
from utils.syntax_checker import (
    validate_code_block,
    check_truncation,
    normalize_line_endings,
    get_common_indent as syntax_get_common_indent,
)

# Compile/lint checking
from utils.compile_checker import CompileChecker, CompileResult, CompileError, check_file_syntax

# Context budget management
from utils.context_budget import ContextBudget

# Input normalization
from utils.input_normalizer import normalize_input

# Logging configuration
from utils.logging import get_logger

# Persistence utilities
from utils.persistence import Persistor

# Prompt loading
from utils.prompt_loader import load_template, safe_format, load_and_format

# RAG query building
from utils.rag_query_builder import RAGQueryBuilder, QueryIntent, CycleAnalysis

# Snippet selection
from utils.snippet_selector import select_relevant_snippet

# Cycle verification
from utils.cycle_verifier import verify_cycle_broken

__all__ = [
    # Patch parsing
    "parse_json_patches",
    "parse_marker_patches",
    "parse_search_replace_json",
    "infer_patches",
    "clean_code_content",
    "extract_patches_from_data",
    "extract_patches_lenient",
    "extract_json_string_value",
    # Patch application
    "SearchReplaceResult",
    "MatchResult",
    "apply_search_replace_atomic",
    "apply_search_replace_list_atomic",
    "apply_with_partial_rollback",
    "try_find_search_text",
    "extract_line_hint",
    # Diff utilities
    "make_unified_diff",
    "looks_truncated",
    "get_common_indent",
    "validate_patched_content",
    "check_for_truncation",
    "compute_diff_stats",
    # Syntax checking
    "validate_code_block",
    "check_truncation",
    "normalize_line_endings",
    # Compile checking
    "CompileChecker",
    "CompileResult",
    "CompileError",
    "check_file_syntax",
    # Context budget
    "ContextBudget",
    # Input normalization
    "normalize_input",
    # Logging
    "get_logger",
    # Persistence
    "Persistor",
    # Prompt loading
    "load_template",
    "safe_format",
    "load_and_format",
    # RAG query building
    "RAGQueryBuilder",
    "QueryIntent",
    "CycleAnalysis",
    # Snippet selection
    "select_relevant_snippet",
    # Cycle verification
    "verify_cycle_broken",
]