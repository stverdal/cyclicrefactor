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


class IOConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    cyclic_folder: str = "cyclicDepen"
    prompts_dir: str = "prompts"


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
    auth: Dict[str, str] = {}
    prompts: Dict[str, str] = {}
    logging: Dict[str, Any] = {}


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
