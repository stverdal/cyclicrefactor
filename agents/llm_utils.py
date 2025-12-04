from typing import Any


def call_llm(llm: Any, prompt: str) -> str:
    """Try common call patterns for different LLM wrappers and return string output.

    Supports objects with methods: invoke(str), predict(str), or simple callables.
    """
    # 1. Try LangChain-style LLMs with string input: llm.invoke(prompt)
    if hasattr(llm, "invoke"):
        try:
            res = llm.invoke(prompt)
            # res may be an AIMessage, dict, or string
            if hasattr(res, "content"):
                # LangChain AIMessage
                return str(res.content)
            if isinstance(res, dict):
                for key in ("answer", "result", "output", "text", "content"):
                    if key in res:
                        return str(res[key])
                return str(res)
            return str(res)
        except Exception as e:
            raise RuntimeError(f"LLM.invoke failed: {e}")

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
    Returns None if the LLM service is not available.
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
        # Check if Ollama is reachable before importing/creating client
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(('localhost', 11434))
            sock.close()
            if result != 0:
                print("[WARNING] Ollama not reachable at localhost:11434, running without LLM")
                return None
        except Exception:
            sock.close()
            print("[WARNING] Could not check Ollama availability, running without LLM")
            return None

        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model, **params)
            return llm
        except Exception as e:
            print(f"[WARNING] Failed to create Ollama client: {e}")
            return None

    # Anthropic (langchain_anthropic.ChatAnthropic)
    if provider and "anthropic" in provider.lower():
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[WARNING] ANTHROPIC_API_KEY not set, running without LLM")
            return None
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model_name=model, **params)
            return llm
        except Exception as e:
            print(f"[WARNING] Failed to create Anthropic client: {e}")
            return None

    # Fallback: return None (no LLM available)
    print(f"[WARNING] Unknown LLM provider '{provider}', running without LLM")
    return None
