from typing import Any


def call_llm(llm: Any, prompt: str) -> str:
    """Try common call patterns for different LLM wrappers and return string output.

    Supports objects with methods: invoke({input:...}), predict(str), or simple callables.
    """
    # Prefer passing a raw string / PromptValue where possible.
    # 1. Try LangChain-style LLMs with string input: llm.invoke(prompt)
    if hasattr(llm, "invoke"):
        try:
            res = llm.invoke(prompt)
        except Exception:
            # Some implementations expect a dict-like payload. Try that as a fallback.
            try:
                res = llm.invoke({"input": prompt})
            except Exception as e:
                raise RuntimeError(f"LLM.invoke failed for both string and dict inputs: {e}")

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
        try:
            return llm.predict(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM.predict failed: {e}")

    # 3. Callable (simple function)
    if callable(llm):
        try:
            return llm(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM callable failed: {e}")

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
