"""Configuration-specific exceptions for paperless-ngx-smart-ocr."""

from __future__ import annotations


__all__ = [
    "ConfigurationError",
    "ConfigurationFileNotFoundError",
    "ConfigurationValidationError",
]


class ConfigurationError(Exception):
    """Base exception for configuration errors.

    All configuration-related exceptions inherit from this class,
    allowing callers to catch all config errors with a single except clause.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
        """
        super().__init__(message)
        self.message = message


class ConfigurationFileNotFoundError(ConfigurationError):
    """Raised when a configuration file cannot be found.

    Attributes:
        path: The path that was requested (may be None if searching defaults).
        searched_paths: List of paths that were searched.
    """

    def __init__(
        self,
        path: str | None = None,
        searched_paths: list[str] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            path: The specific path requested, or None if searching defaults.
            searched_paths: List of paths that were searched.
        """
        self.path = path
        self.searched_paths = searched_paths or []

        if self.searched_paths:
            paths_str = ", ".join(self.searched_paths)
            message = f"Configuration file not found. Searched: {paths_str}"
        elif path:
            message = f"Configuration file not found: {path}"
        else:
            message = "Configuration file not found"

        super().__init__(message)


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails.

    This exception wraps Pydantic validation errors with a more
    user-friendly message and provides access to the detailed errors.

    Attributes:
        errors: List of validation error details from Pydantic.
    """

    def __init__(
        self,
        message: str,
        errors: list[dict[str, object]] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable summary of the validation failure.
            errors: List of validation error details (from Pydantic).
        """
        super().__init__(message)
        self.errors = errors or []
