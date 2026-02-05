"""Custom exceptions for the Paperless-ngx API client."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import httpx


__all__ = [
    "PaperlessAuthenticationError",
    "PaperlessConnectionError",
    "PaperlessError",
    "PaperlessNotFoundError",
    "PaperlessRateLimitError",
    "PaperlessServerError",
    "PaperlessValidationError",
]


class PaperlessError(Exception):
    """Base exception for all Paperless-ngx client errors.

    Attributes:
        message: Human-readable error description.
        response: The HTTP response that caused this error, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            response: The HTTP response that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self) -> str:
        """Return string representation with status code if available."""
        if self.response is not None:
            return f"{self.message} (status={self.response.status_code})"
        return self.message


class PaperlessConnectionError(PaperlessError):
    """Raised when connection to Paperless-ngx fails.

    This includes network errors, DNS failures, and timeouts.
    These errors are typically retryable.
    """

    def __init__(
        self,
        message: str = "Failed to connect to Paperless-ngx",
        *,
        cause: Exception | None = None,
    ) -> None:
        """Initialize the connection error.

        Args:
            message: Human-readable error description.
            cause: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.__cause__ = cause


class PaperlessAuthenticationError(PaperlessError):
    """Raised for authentication failures (401/403).

    This indicates invalid or expired API tokens, or insufficient permissions.
    These errors are not retryable without fixing the credentials.
    """


class PaperlessNotFoundError(PaperlessError):
    """Raised when a resource is not found (404).

    Attributes:
        resource_type: The type of resource that was not found (e.g., "Document").
        resource_id: The ID of the resource that was not found.
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: int | str,
        *,
        response: httpx.Response | None = None,
    ) -> None:
        """Initialize the not found error.

        Args:
            resource_type: The type of resource (e.g., "Document", "Tag").
            resource_id: The ID that was not found.
            response: The HTTP response that caused this error.
        """
        message = f"{resource_type} with id={resource_id} not found"
        super().__init__(message, response=response)
        self.resource_type = resource_type
        self.resource_id = resource_id


class PaperlessRateLimitError(PaperlessError):
    """Raised when rate limited by Paperless-ngx (429).

    Attributes:
        retry_after: Number of seconds to wait before retrying, if provided.
    """

    def __init__(
        self,
        message: str = "Rate limited by Paperless-ngx",
        *,
        retry_after: float | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Human-readable error description.
            retry_after: Seconds to wait before retrying.
            response: The HTTP response that caused this error.
        """
        super().__init__(message, response=response)
        self.retry_after = retry_after


class PaperlessServerError(PaperlessError):
    """Raised for server errors (5xx).

    These indicate problems on the Paperless-ngx server side.
    These errors may be retryable after a backoff period.
    """


class PaperlessValidationError(PaperlessError):
    """Raised for validation errors (400).

    This indicates the request was malformed or contained invalid data.

    Attributes:
        errors: Dictionary mapping field names to lists of error messages.
    """

    def __init__(
        self,
        message: str,
        *,
        errors: dict[str, list[str]] | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        """Initialize the validation error.

        Args:
            message: Human-readable error description.
            errors: Field-level validation errors.
            response: The HTTP response that caused this error.
        """
        super().__init__(message, response=response)
        self.errors = errors or {}
