from typing import Any
from utils.logging import get_logger

logger = get_logger("llm_utils")


def call_llm(llm: Any, prompt: str) -> str:
    """Try common call patterns for different LLM wrappers and return string output.

    Supports objects with methods: invoke(str), predict(str), or simple callables.
    """
    logger.debug(f"call_llm invoked with prompt length: {len(prompt)} chars")

    # 1. Try LangChain-style LLMs with string input: llm.invoke(prompt)
    if hasattr(llm, "invoke"):
        try:
            logger.debug("Attempting LLM.invoke()")
            res = llm.invoke(prompt)
            # res may be an AIMessage, dict, or string
            if hasattr(res, "content"):
                # LangChain AIMessage
                content = str(res.content)
                logger.debug(f"LLM returned AIMessage, content length: {len(content)} chars")
                return content
            if isinstance(res, dict):
                for key in ("answer", "result", "output", "text", "content"):
                    if key in res:
                        logger.debug(f"LLM returned dict with '{key}' key")
                        return str(res[key])
                return str(res)
            logger.debug(f"LLM returned raw value, length: {len(str(res))} chars")
            return str(res)
        except Exception as e:
            logger.error(f"LLM.invoke failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM.invoke failed: {e}")

    # 2. Objects with `predict` (some wrappers)
    if hasattr(llm, "predict"):
        try:
            logger.debug("Attempting LLM.predict()")
            return llm.predict(prompt)
        except Exception as e:
            logger.error(f"LLM.predict failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM.predict failed: {e}")

    # 3. Callable (simple function)
    if callable(llm):
        try:
            logger.debug("Attempting LLM as callable")
            return llm(prompt)
        except Exception as e:
            logger.error(f"LLM callable failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM callable failed: {e}")

    logger.error("LLM object has no supported call interface")
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
        logger.debug("No LLM config provided")
        return None
    if isinstance(cfg, dict):
        provider = cfg.get("provider")
        model = cfg.get("model")
        params = cfg.get("params", {})
    else:
        provider = getattr(cfg, "provider", None)
        model = getattr(cfg, "model", None)
        params = getattr(cfg, "params", {}) or {}

    logger.info(f"Creating LLM client: provider={provider}, model={model}")

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
                logger.warning("Ollama not reachable at localhost:11434, running without LLM")
                return None
        except Exception:
            sock.close()
            logger.warning("Could not check Ollama availability, running without LLM")
            return None

        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=model, **params)
            logger.info(f"Ollama client created successfully for model: {model}")
            return llm
        except Exception as e:
            logger.warning(f"Failed to create Ollama client: {e}")
            return None

    # NOTE: External API providers (Anthropic, OpenAI) removed for air-gapped environment
    # Only local providers (Ollama) are supported

    # Fallback: return None (no LLM available)
    logger.warning(f"Unknown LLM provider '{provider}', running without LLM")
    return None
