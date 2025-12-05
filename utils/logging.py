"""Logging utilities for the cycle-refactoring pipeline.

Provides a centralized logging setup that can be configured via config.yml.
"""

import logging
import sys
from typing import Optional, Dict, Any

# Global reference to the root logger for the pipeline
_pipeline_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "cycle_refactoring",
    log_file: str = "langCodeUnderstanding.log",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with file and optional console handlers.

    Args:
        name: Logger name (module name or custom).
        log_file: Path to log file.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to also log to console.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(console_handler)

    return logger


def configure_from_config(config: Dict[str, Any]) -> logging.Logger:
    """Configure the pipeline logger from config dict.

    Args:
        config: Dict with 'level' and 'log_file' keys.

    Returns:
        Configured root pipeline logger.
    """
    global _pipeline_logger

    level = config.get("level", "INFO")
    log_file = config.get("log_file", "langCodeUnderstanding.log")
    console = config.get("console", True)

    _pipeline_logger = setup_logger(
        name="cycle_refactoring",
        log_file=log_file,
        level=level,
        console=console,
    )
    return _pipeline_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger for a specific module.

    If the pipeline logger has been configured, child loggers will inherit
    its handlers and level.

    Args:
        name: Module or component name (e.g., "describer", "validator").

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"cycle_refactoring.{name}")
    return logging.getLogger("cycle_refactoring")
