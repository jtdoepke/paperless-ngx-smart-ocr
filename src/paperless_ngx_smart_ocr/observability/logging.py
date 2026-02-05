"""Structured logging configuration for paperless-ngx-smart-ocr.

This module provides a human-readable, machine-parseable logging setup using
structlog with logfmt-style output. It supports:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Request ID tracking via contextvars for correlation
- Colorized console output for development (TTY detection)
- ISO 8601 timestamps in UTC
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)
from structlog.processors import TimeStamper, add_log_level


if TYPE_CHECKING:
    from collections.abc import Sequence

    from structlog.typing import EventDict, WrappedLogger

__all__ = [
    "LogLevel",
    "clear_request_context",
    "configure_logging",
    "generate_request_id",
    "get_logger",
    "get_request_id",
    "set_request_id",
]


class LogLevel(StrEnum):
    """Supported log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def to_stdlib_level(self) -> int:
        """Convert to stdlib logging level.

        Returns:
            The corresponding logging module level constant.
        """
        level: int = getattr(logging, self.name)
        return level


# Context variable for request ID tracking
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def generate_request_id() -> str:
    """Generate a new unique request ID.

    Returns:
        A short UUID-based request ID (first 8 characters).
    """
    return uuid.uuid4().hex[:8]


def get_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        The current request ID, or None if not set.
    """
    return _request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set the request ID in context.

    If no request_id is provided, a new one is generated.
    Also binds the request_id to structlog's contextvars.

    Args:
        request_id: Optional request ID to set. If None, generates a new one.

    Returns:
        The request ID that was set.
    """
    if request_id is None:
        request_id = generate_request_id()

    _request_id_var.set(request_id)
    bind_contextvars(request_id=request_id)
    return request_id


def clear_request_context() -> None:
    """Clear the request context (request ID and structlog contextvars).

    Should be called at the start of each request to ensure clean state.
    """
    _request_id_var.set(None)
    clear_contextvars()


def add_request_id(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add request_id to event dict if present in context and not already set.

    This processor ensures request_id is included even if merge_contextvars
    hasn't captured it yet.

    Args:
        logger: The wrapped logger object (unused but required by protocol).
        method_name: The name of the log method called (unused but required).
        event_dict: The event dictionary to process.

    Returns:
        The event dictionary with request_id added if available.
    """
    del logger, method_name  # Unused but required by processor protocol
    if "request_id" not in event_dict:
        request_id = get_request_id()
        if request_id is not None:
            event_dict["request_id"] = request_id
    return event_dict


def _create_renderer(
    *,
    colors: bool = True,
    key_order: Sequence[str] | None = None,
) -> structlog.dev.ConsoleRenderer | structlog.processors.LogfmtRenderer:
    """Create appropriate renderer based on environment.

    When colors are enabled (TTY detected), uses ConsoleRenderer for
    pretty output. Otherwise, uses LogfmtRenderer for machine-parseable output.

    Args:
        colors: Whether to enable colorized output.
        key_order: Order of keys in logfmt output.

    Returns:
        Configured renderer processor.
    """
    if colors:
        # Use ConsoleRenderer for colorized, human-friendly output
        return structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    # Use LogfmtRenderer for machine-parseable output
    default_key_order = ["timestamp", "level", "event", "request_id"]
    order = list(key_order) if key_order else default_key_order
    return structlog.processors.LogfmtRenderer(
        key_order=order,
        drop_missing=True,
        bool_as_flag=False,  # Use explicit true/false for machines
    )


def configure_logging(
    level: LogLevel | str = LogLevel.INFO,
    *,
    force_colors: bool | None = None,
) -> None:
    """Configure structured logging for the application.

    Sets up structlog with:
    - Logfmt-style output (human and machine readable)
    - ISO 8601 timestamps in UTC
    - Log level names
    - Request ID from contextvars
    - Optional colorized output (auto-detected from TTY)

    This function should be called once at application startup,
    typically in the CLI or web app initialization.

    Args:
        level: Minimum log level. Can be a LogLevel enum or string
            ('debug', 'info', 'warning', 'error', 'critical').
        force_colors: Force color output on/off. If None, auto-detect from TTY.

    Example:
        >>> from paperless_ngx_smart_ocr.observability import (
        ...     configure_logging,
        ...     LogLevel,
        ... )
        >>> configure_logging(level=LogLevel.DEBUG)
        >>> # Or with string:
        >>> configure_logging(level="info")
    """
    # Normalize level to LogLevel enum
    if isinstance(level, str):
        level = LogLevel(level.lower())

    # Determine if we should use colors
    if force_colors is not None:
        use_colors = force_colors
    else:
        # Auto-detect: use colors if stderr is a TTY
        use_colors = (
            sys.stderr is not None
            and hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
        )

    # Build processor chain
    # Order matters! Each processor transforms the event dict for the next.
    processors: list[structlog.typing.Processor] = [
        # Merge context from contextvars (e.g., request_id)
        merge_contextvars,
        # Add our request_id if not already present
        add_request_id,
        # Add log level name (debug, info, warning, error, critical)
        add_log_level,
        # Add ISO 8601 timestamp in UTC
        TimeStamper(fmt="iso", utc=True, key="timestamp"),
        # Render stack info if present
        structlog.processors.StackInfoRenderer(),
        # Handle exc_info
        structlog.dev.set_exc_info,
        # Final renderer (Console or Logfmt based on TTY)
        _create_renderer(colors=use_colors),
    ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level.to_stdlib_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for libraries that use it
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level.to_stdlib_level(),
        force=True,
    )


def get_logger(
    name: str | None = None,
    **initial_context: object,
) -> structlog.BoundLogger:
    """Get a structured logger instance.

    This is the primary way to obtain a logger in the application.
    The logger automatically includes any context bound via contextvars
    (like request_id).

    Args:
        name: Logger name, typically __name__ of the calling module.
        **initial_context: Initial key-value pairs to bind to the logger.

    Returns:
        A bound structlog logger.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("processing_document", document_id=123)
        2024-01-15T10:30:45.123456Z [info] processing_document document_id=123

        >>> # With initial context
        >>> logger = get_logger(__name__, component="pipeline")
        >>> logger.info("starting")
        2024-01-15T10:30:45.123456Z [info] starting component=pipeline
    """
    log: structlog.BoundLogger = structlog.get_logger(name)
    if initial_context:
        log = log.bind(**initial_context)
    return log
