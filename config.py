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


class IOConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    cyclic_folder: str = "cyclicDepen"
    prompts_dir: str = "prompts"


class PipelineConfig(BaseModel):
    agents_order: list = ["describer", "refactor", "validator"]
    max_iterations: int = 1
    enable: Dict[str, bool] = {"describer": True, "refactor": True, "validator": True}


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()
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
