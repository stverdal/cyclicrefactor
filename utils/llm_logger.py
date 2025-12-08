"""Enhanced LLM Input/Output logging utilities.

This module provides specialized logging for LLM interactions:
- Separate log file for LLM prompts and responses
- Timestamps and context for each interaction
- Configurable truncation for large prompts/responses
- Structured format for later analysis
"""

import logging
import json
from datetime import datetime
from typing import Optional, Any, Dict
from pathlib import Path

# Module-level logger for LLM I/O
_llm_io_logger: Optional[logging.Logger] = None
_llm_io_config: Dict[str, Any] = {}


def setup_llm_io_logger(
    log_file: str = "llm_io.log",
    log_prompts: bool = True,
    log_responses: bool = True,
    truncate_at: int = 0,
    log_timestamps: bool = True,
) -> logging.Logger:
    """Set up dedicated LLM I/O logger.
    
    Args:
        log_file: Path to log file for LLM I/O
        log_prompts: Whether to log prompts sent to LLM
        log_responses: Whether to log LLM responses
        truncate_at: Max characters to log (0 = no truncation)
        log_timestamps: Whether to include timestamps
    
    Returns:
        Configured logger for LLM I/O
    """
    global _llm_io_logger, _llm_io_config
    
    _llm_io_config = {
        "log_prompts": log_prompts,
        "log_responses": log_responses,
        "truncate_at": truncate_at,
        "log_timestamps": log_timestamps,
    }
    
    logger = logging.getLogger("llm_io")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    if log_timestamps:
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    
    # File handler for LLM I/O
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create LLM I/O log file: {e}")
    
    # Don't propagate to root logger (avoid duplicate console output)
    logger.propagate = False
    
    _llm_io_logger = logger
    return logger


def get_llm_io_logger() -> Optional[logging.Logger]:
    """Get the LLM I/O logger if configured."""
    return _llm_io_logger


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text if max_chars is set and text is too long."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [TRUNCATED - {len(text) - max_chars} more chars]"


def log_llm_call(
    agent_name: str,
    stage: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
) -> str:
    """Log an LLM prompt being sent.
    
    Args:
        agent_name: Name of the agent making the call (e.g., "refactor", "describer")
        stage: Pipeline stage or purpose (e.g., "planning", "file_patch", "validation")
        prompt: The full prompt being sent to the LLM
        context: Optional additional context dict
        call_id: Optional unique ID for this call (for correlating with response)
    
    Returns:
        call_id for correlating with response log
    """
    if _llm_io_logger is None:
        return call_id or ""
    
    if not _llm_io_config.get("log_prompts", True):
        return call_id or ""
    
    # Generate call ID if not provided
    if call_id is None:
        call_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Truncate if configured
    truncate_at = _llm_io_config.get("truncate_at", 0)
    prompt_to_log = _truncate(prompt, truncate_at)
    
    # Build log entry
    separator = "=" * 80
    entry_lines = [
        separator,
        f"LLM CALL | Agent: {agent_name} | Stage: {stage} | ID: {call_id}",
        f"Prompt Length: {len(prompt)} chars",
    ]
    
    if context:
        entry_lines.append(f"Context: {json.dumps(context, default=str)}")
    
    entry_lines.extend([
        "-" * 40 + " PROMPT " + "-" * 40,
        prompt_to_log,
        separator,
    ])
    
    _llm_io_logger.info("\n".join(entry_lines))
    
    return call_id


def log_llm_response(
    agent_name: str,
    stage: str,
    response: Any,
    call_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log an LLM response received.
    
    Args:
        agent_name: Name of the agent that made the call
        stage: Pipeline stage or purpose
        response: The LLM response (string or dict)
        call_id: ID from the corresponding log_llm_call
        duration_ms: Time taken for the LLM call in milliseconds
        error: Error message if the call failed
    """
    if _llm_io_logger is None:
        return
    
    if not _llm_io_config.get("log_responses", True):
        return
    
    # Convert response to string
    if isinstance(response, str):
        response_text = response
    elif isinstance(response, dict):
        try:
            response_text = json.dumps(response, indent=2, default=str)
        except Exception:
            response_text = str(response)
    else:
        response_text = str(response)
    
    # Truncate if configured
    truncate_at = _llm_io_config.get("truncate_at", 0)
    response_to_log = _truncate(response_text, truncate_at)
    
    # Build log entry
    separator = "=" * 80
    entry_lines = [
        separator,
        f"LLM RESPONSE | Agent: {agent_name} | Stage: {stage} | ID: {call_id or 'unknown'}",
        f"Response Length: {len(response_text)} chars",
    ]
    
    if duration_ms is not None:
        entry_lines.append(f"Duration: {duration_ms:.0f}ms")
    
    if error:
        entry_lines.append(f"ERROR: {error}")
    
    entry_lines.extend([
        "-" * 40 + " RESPONSE " + "-" * 38,
        response_to_log,
        separator,
    ])
    
    _llm_io_logger.info("\n".join(entry_lines))


def log_llm_summary(
    agent_name: str,
    total_calls: int,
    successful_calls: int,
    total_prompt_chars: int,
    total_response_chars: int,
    total_duration_ms: float,
) -> None:
    """Log a summary of LLM calls for an agent.
    
    Args:
        agent_name: Name of the agent
        total_calls: Total number of LLM calls made
        successful_calls: Number of successful calls
        total_prompt_chars: Total characters in all prompts
        total_response_chars: Total characters in all responses
        total_duration_ms: Total time spent on LLM calls
    """
    if _llm_io_logger is None:
        return
    
    separator = "*" * 80
    entry_lines = [
        separator,
        f"LLM SUMMARY | Agent: {agent_name}",
        f"Calls: {successful_calls}/{total_calls} successful",
        f"Total Prompt Chars: {total_prompt_chars:,}",
        f"Total Response Chars: {total_response_chars:,}",
        f"Total Duration: {total_duration_ms:.0f}ms ({total_duration_ms/1000:.1f}s)",
        f"Avg Per Call: {total_duration_ms/total_calls:.0f}ms" if total_calls > 0 else "N/A",
        separator,
    ]
    
    _llm_io_logger.info("\n".join(entry_lines))


class LLMCallTracker:
    """Context manager for tracking LLM calls with automatic logging.
    
    Usage:
        with LLMCallTracker("refactor", "planning") as tracker:
            tracker.log_prompt(prompt)
            response = call_llm(llm, prompt)
            tracker.log_response(response)
    """
    
    def __init__(self, agent_name: str, stage: str, context: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        self.stage = stage
        self.context = context
        self.call_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.prompt: Optional[str] = None
        self.response: Optional[Any] = None
        self.error: Optional[str] = None
    
    def __enter__(self) -> "LLMCallTracker":
        self.start_time = datetime.now()
        self.call_id = self.start_time.strftime("%Y%m%d_%H%M%S_%f")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.error = str(exc_val)
    
    def log_prompt(self, prompt: str) -> None:
        """Log the prompt being sent."""
        self.prompt = prompt
        log_llm_call(
            self.agent_name,
            self.stage,
            prompt,
            context=self.context,
            call_id=self.call_id,
        )
    
    def log_response(self, response: Any) -> None:
        """Log the response received."""
        self.response = response
        duration_ms = None
        if self.start_time:
            duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        log_llm_response(
            self.agent_name,
            self.stage,
            response,
            call_id=self.call_id,
            duration_ms=duration_ms,
            error=self.error,
        )


def configure_from_config(config: Dict[str, Any]) -> None:
    """Configure LLM I/O logging from config dict.
    
    Args:
        config: Dict with logging configuration
    """
    if not config.get("log_llm_io", True):
        return
    
    setup_llm_io_logger(
        log_file=config.get("llm_io_log_file", "llm_io.log"),
        log_prompts=config.get("log_llm_prompts", True),
        log_responses=config.get("log_llm_responses", True),
        truncate_at=config.get("truncate_llm_logs", 0),
        log_timestamps=config.get("log_llm_timestamps", True),
    )
