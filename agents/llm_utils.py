from typing import Any


def call_llm(llm: Any, prompt: str) -> str:
    """Try common call patterns for different LLM wrappers and return string output.

    Supports objects with methods: invoke({input:...}), predict(str), or simple callables.
    """
    # 1. LangChain-style LLMs that expose `invoke`
    if hasattr(llm, "invoke"):
        res = llm.invoke({"input": prompt})
        # res may be dict-like or string
        if isinstance(res, dict):
            for key in ("answer", "result", "output", "text"):
                if key in res:
                    return res[key]
            # Fallback to stringifying
            return str(res)
        return str(res)

    # 2. Objects with `predict` (some wrappers)
    if hasattr(llm, "predict"):
        return llm.predict(prompt)

    # 3. Callable (simple function)
    if callable(llm):
        return llm(prompt)

    raise RuntimeError("LLM object has no supported call interface (invoke/predict/callable)")


def create_llm_from_config(cfg: Any) -> Any:
    """Factory: create an LLM client from a config-like object.

    cfg can be a dict-like or object with attributes `provider`, `model`, `params`.
    The returned object should be compatible with `call_llm` (invoke/predict/callable).
    """
    provider = None
    model = None
    params = {}
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        provider = cfg.get("provider")
        model = cfg.get("model")
        params = cfg.get("params", {})
    else:
        provider = getattr(cfg, "provider", None)
        model = getattr(cfg, "model", None)
        params = getattr(cfg, "params", {}) or {}

    # Ollama (langchain_ollama.ChatOllama)
    if provider and "ollama" in provider.lower():
        try:
            from langchain_ollama import ChatOllama

            # Map common params
            llm = ChatOllama(model=model, **params)
            return llm
        except Exception:
            # Fallback mock
            return lambda prompt: f"[mock-ollama] {prompt}"

    # Anthropic (langchain_anthropic.ChatAnthropic)
    if provider and "anthropic" in provider.lower():
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model_name=model, **params)
            return llm
        except Exception:
            return lambda prompt: f"[mock-anthropic] {prompt}"

    # Fallback: return a simple echo callable
    return lambda prompt: f"[mock-llm] {prompt}"
