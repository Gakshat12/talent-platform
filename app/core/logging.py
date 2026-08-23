"""Structured logging configuration module using Loguru for the AI Talent Intelligence Platform."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Format specified by system requirements
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


def configure_logging() -> None:
    """Configures Loguru handlers for standard error output and rotating file log.

    - Standard error handler with level INFO.
    - File handler writing to logs/app.log with 10 MB rotation and 7 days retention.
    """
    logger.remove()

    # Add stderr handler
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level="INFO",
        colorize=False,
    )

    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add rotating file handler
    log_file_path = log_dir / "app.log"
    logger.add(
        str(log_file_path),
        format=LOG_FORMAT,
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
    )


def get_logger(name: str) -> Any:
    """Returns a logger instance bound with the specific module name context.

    Args:
        name: The name of the calling module, typically __name__.

    Returns:
        Bound Loguru logger object.
    """
    return logger.bind(name=name)


# Automatically configure logging when the module is imported
configure_logging()
