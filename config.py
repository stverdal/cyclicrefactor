from typing import Any, Dict, Optional
from pydantic import BaseModel
import yaml
import os


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5-coder:32b-32768"
    params: Dict[str, Any] = {}


class RetrieverConfig(BaseModel):
    type: str = "chroma"
    persist_dir: str = "cache/chroma_db"
    data_dir: str = "data/pdf"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    collection_name: str = "architecture_docs"
    search_type: str = "similarity"  # "similarity" or "mmr"
    search_kwargs: Dict[str, Any] = {"k": 4}


class RefactorConfig(BaseModel):
    """Configuration for the RefactorAgent's SEARCH/REPLACE matching behavior."""
    min_match_confidence: float = 0.7  # Minimum confidence to accept a match
    warn_confidence: float = 0.85       # Warn if confidence below this (but still apply)
    allow_low_confidence: bool = False  # If True, accept matches below min_confidence with warning
    atomic_mode: bool = True            # If True, all SEARCH/REPLACE blocks must succeed or none applied
    atomic_proposal: bool = True        # If True, all files in proposal must succeed or all are reverted
    revert_on_any_critical: bool = True # If True, revert entire proposal if any file has critical errors
    compile_check: bool = False         # If True, run language-specific compile/lint check on patches
    compile_check_timeout: int = 30     # Timeout in seconds for compile commands
    revert_on_compile_error: bool = True  # If True, revert files that fail compile check
    compact_prompts: bool = False       # If True, use shorter prompts for low-VRAM/small context LLMs
    auto_compact_threshold: int = 8192  # Auto-enable compact prompts if context window below this
    # Sequential file mode - process files one at a time
    sequential_file_mode: bool = False  # If True, use two-phase approach: plan then per-file execution
    auto_sequential_threshold: int = 3  # Auto-enable sequential mode if cycle has >= this many files
    # Auto-repair settings
    auto_repair_syntax: bool = True     # If True, attempt to auto-repair syntax errors in LLM output
    auto_repair_min_confidence: float = 0.6  # Minimum confidence to accept auto-repair
    # Validation settings
    rule_based_validation: bool = True  # If True, run rule-based validation (set False to skip)
    block_on_validation_failure: bool = True  # If True, reject patches that fail validation
    hallucination_detection: bool = True  # If True, detect and warn about LLM hallucinations
    # === NEW: Advanced Refactoring Modes ===
    # Roadmap mode - outputs detailed progress even on failures (good for demos)
    roadmap_mode: bool = False          # If True, output RefactorRoadmap with partial results
    # Minimal diff mode - focus on single smallest change to break cycle
    minimal_diff_mode: bool = False     # If True, generate only the minimal change needed
    # Scaffolding mode - create interfaces/abstractions BEFORE modifying existing files
    scaffolding_mode: bool = False      # If True, create new files first, validate, then modify
    scaffolding_validate: bool = True   # If True, compile-check scaffold files before proceeding
    # Full file context - send complete files when they fit in context
    prioritize_full_files: bool = True  # If True, send full file content when it fits
    full_file_max_chars: int = 8000     # Max chars per file to include full content
    full_file_budget_pct: float = 0.4   # Max % of context window for file content
    # Simple format mode - use line-based format instead of JSON for very small LLMs
    simple_format_mode: bool = False    # If True, use simple line-based format for 7B and smaller
    auto_simple_threshold: int = 8      # Auto-enable simple format if model size (B) <= this
    # Suggestion mode - output suggestions for human review instead of applying patches
    suggestion_mode: bool = False       # If True, output suggestions without applying patches
    suggestion_context_lines: int = 7   # Number of context lines to include around changes
    suggestion_output_format: str = "markdown"  # Output format: "markdown" or "json"
    # Relaxed suggestion mode - simple prompt, human-readable output, no strict validation
    relaxed_suggestion_mode: bool = False  # If True, use simple prompt and skip strict parsing
    relaxed_skip_validation: bool = True   # If True, skip validation entirely in relaxed mode
    # Line-based patching - use line numbers instead of SEARCH/REPLACE text matching
    line_based_patching: bool = False   # If True, use line numbers for more reliable patching
    line_patch_backup: bool = True      # If True, create backup before applying line patches


class IOConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    cyclic_folder: str = "cyclicDepen"
    prompts_dir: str = "prompts"


class LoggingConfig(BaseModel):
    """Configuration for enhanced logging."""
    level: str = "INFO"
    log_file: str = "langCodeUnderstanding.log"
    console: bool = True
    # LLM input/output logging
    log_llm_io: bool = True          # If True, log LLM inputs and outputs
    llm_io_log_file: str = "llm_io.log"  # Separate file for LLM I/O logs
    log_llm_prompts: bool = True     # If True, log full prompts sent to LLM
    log_llm_responses: bool = True   # If True, log full LLM responses
    truncate_llm_logs: int = 0       # Max chars to log per prompt/response (0 = no truncation)
    log_llm_timestamps: bool = True  # If True, include timestamps in LLM logs


class PipelineConfig(BaseModel):
    agents_order: list = ["describer", "refactor", "validator", "explainer"]
    max_iterations: int = 1
    enable: Dict[str, bool] = {"describer": True, "refactor": True, "validator": True, "explainer": True}
    dry_run: bool = False  # If True, run pipeline without writing any files
    dry_run_log_writes: bool = True  # If True, log what would be written in dry-run mode


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    refactor: RefactorConfig = RefactorConfig()
    io: IOConfig = IOConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()
    auth: Dict[str, str] = {}
    prompts: Dict[str, str] = {}


def load_config(path: Optional[str] = None) -> AppConfig:
    path = path or os.environ.get("CYCLE_REFACTOR_CONFIG", "config.yml")
    if not os.path.exists(path):
        # Return defaults if no config file
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Merge env-based secrets if indicated (simple pattern)
    auth = raw.get("auth", {})
    for k, v in auth.items():
        if isinstance(v, str) and v.endswith("_ENV"):
            # Example: {some_key_env: SOME_ENV_VAR}
            env_name = raw["auth"].get(k)
            raw["auth"][k] = os.environ.get(env_name, None)

    return AppConfig(**raw)
