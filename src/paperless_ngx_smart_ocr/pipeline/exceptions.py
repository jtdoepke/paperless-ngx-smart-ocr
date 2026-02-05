"""Pipeline-specific exceptions.

This module defines exceptions raised during document processing,
including preprocessing, layout detection, and OCR operations.
"""

from __future__ import annotations


__all__ = [
    "LayoutDetectionError",
    "OCRError",
    "PipelineError",
    "PreprocessingError",
]


class PipelineError(Exception):
    """Base exception for all pipeline errors.

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

    def __str__(self) -> str:
        """Return string representation."""
        return self.message


class PreprocessingError(PipelineError):
    """Error during document preprocessing or analysis.

    Raised when document analysis fails, such as when a PDF cannot be
    opened, is corrupted, or cannot be analyzed for text content.

    Attributes:
        message: Human-readable error description.
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            cause: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class LayoutDetectionError(PipelineError):
    """Error during Surya layout detection.

    Raised when layout detection fails for a document or page.

    Attributes:
        message: Human-readable error description.
        page: The page number where the error occurred (0-indexed).
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        *,
        page: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            page: The page number where the error occurred (0-indexed).
            cause: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.page = page
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return string representation with page number if available."""
        if self.page is not None:
            return f"{self.message} (page={self.page})"
        return self.message


class OCRError(PipelineError):
    """Error during OCR processing with OCRmyPDF.

    Raised when OCRmyPDF fails to process a document.

    Attributes:
        message: Human-readable error description.
        exit_code: The OCRmyPDF exit code, if available.
        cause: The underlying exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            exit_code: The OCRmyPDF exit code, if available.
            cause: The underlying exception that caused this error.
        """
        super().__init__(message)
        self.exit_code = exit_code
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return string representation with exit code if available."""
        if self.exit_code is not None:
            return f"{self.message} (exit_code={self.exit_code})"
        return self.message
