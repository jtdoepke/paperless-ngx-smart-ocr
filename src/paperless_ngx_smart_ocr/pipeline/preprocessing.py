"""Document preprocessing and analysis.

This module provides functions for analyzing PDF documents before processing,
including born-digital detection (checking for existing text layers).

Born-digital documents are PDFs created from digital sources (Word, LaTeX, etc.)
rather than scanned paper documents. These typically already have accurate text
layers and may not need OCR processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fitz  # PyMuPDF

from paperless_ngx_smart_ocr.observability import get_logger
from paperless_ngx_smart_ocr.pipeline.exceptions import PreprocessingError
from paperless_ngx_smart_ocr.pipeline.models import DocumentAnalysis, PageAnalysis


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "analyze_document",
    "has_text_layer",
]


# Threshold for considering a page as having substantial text
_SUBSTANTIAL_TEXT_CHARS = 100


def has_text_layer(pdf_path: Path, *, min_chars: int = 10) -> bool:
    """Check if a PDF has a text layer with meaningful content.

    Examines the PDF to determine if it contains extractable text,
    which indicates either a born-digital document or one that has
    already been OCR'd.

    Args:
        pdf_path: Path to the PDF file to analyze.
        min_chars: Minimum number of non-whitespace characters required
            to consider the document as having text.

    Returns:
        True if the PDF has extractable text with at least min_chars
        characters, False otherwise.

    Raises:
        PreprocessingError: If the file cannot be opened or analyzed.

    Example:
        ```python
        if has_text_layer(pdf_path):
            print("Document already has text, OCR may be unnecessary")
        else:
            print("Document needs OCR")
        ```
    """
    logger = get_logger(__name__)
    logger.debug("checking_text_layer", path=str(pdf_path), min_chars=min_chars)

    try:
        with fitz.open(pdf_path) as doc:
            total_chars = 0
            for page in doc:
                text = page.get_text()
                # Count non-whitespace characters
                chars = len(text.strip())
                total_chars += chars

                # Early exit if we've found enough text
                if total_chars >= min_chars:
                    logger.debug(
                        "text_layer_found",
                        path=str(pdf_path),
                        chars_found=total_chars,
                    )
                    return True

            logger.debug(
                "no_text_layer",
                path=str(pdf_path),
                total_chars=total_chars,
            )
            return False

    except fitz.FileDataError as exc:
        msg = f"Invalid or corrupted PDF file: {pdf_path}"
        raise PreprocessingError(msg, cause=exc) from exc
    except fitz.EmptyFileError as exc:
        msg = f"Empty PDF file: {pdf_path}"
        raise PreprocessingError(msg, cause=exc) from exc
    except Exception as exc:
        msg = f"Failed to analyze PDF for text layer: {exc}"
        raise PreprocessingError(msg, cause=exc) from exc


def analyze_document(pdf_path: Path) -> DocumentAnalysis:
    """Analyze a PDF document for text content and structure.

    Performs a comprehensive analysis of each page to determine:
    - Whether it has extractable text
    - Character count per page
    - Whether pages are image-only (scanned)
    - Overall born-digital classification

    A document is considered "born-digital" if more than 50% of its pages
    have substantial text content (>100 characters), indicating it was
    likely created from digital sources rather than scanned.

    Args:
        pdf_path: Path to the PDF file to analyze.

    Returns:
        DocumentAnalysis containing per-page and document-level statistics.

    Raises:
        PreprocessingError: If the file cannot be opened or analyzed.

    Example:
        ```python
        analysis = analyze_document(pdf_path)
        if analysis.is_born_digital:
            print(f"Born-digital document with {analysis.total_pages} pages")
        else:
            pages_needing_ocr = len(analysis.pages_without_text)
            print(f"Scanned document, {pages_needing_ocr} pages need OCR")
        ```
    """
    logger = get_logger(__name__)
    logger.debug("analyzing_document", path=str(pdf_path))

    try:
        with fitz.open(pdf_path) as doc:
            pages: list[PageAnalysis] = []
            total_text_chars = 0

            for page_num, page in enumerate(doc):
                page_analysis = _analyze_page(page, page_num)
                pages.append(page_analysis)
                total_text_chars += page_analysis.text_char_count

            # Determine if document is born-digital
            # A document is born-digital if >50% of pages have substantial text
            pages_with_substantial_text = sum(
                1 for p in pages if p.text_char_count > _SUBSTANTIAL_TEXT_CHARS
            )
            has_any_text = total_text_chars > 0
            is_born_digital = (
                has_any_text
                and len(pages) > 0
                and pages_with_substantial_text > len(pages) * 0.5
            )

            result = DocumentAnalysis(
                total_pages=len(pages),
                pages=pages,
                has_any_text=has_any_text,
                is_born_digital=is_born_digital,
            )

            logger.info(
                "document_analyzed",
                path=str(pdf_path),
                total_pages=result.total_pages,
                has_any_text=result.has_any_text,
                is_born_digital=result.is_born_digital,
                pages_with_text=len(result.pages_with_text),
                total_chars=total_text_chars,
            )

            return result

    except fitz.FileDataError as exc:
        msg = f"Invalid or corrupted PDF file: {pdf_path}"
        raise PreprocessingError(msg, cause=exc) from exc
    except fitz.EmptyFileError as exc:
        msg = f"Empty PDF file: {pdf_path}"
        raise PreprocessingError(msg, cause=exc) from exc
    except Exception as exc:
        msg = f"Failed to analyze document: {exc}"
        raise PreprocessingError(msg, cause=exc) from exc


def _analyze_page(page: fitz.Page, page_num: int) -> PageAnalysis:
    """Analyze a single PDF page for text content.

    Args:
        page: PyMuPDF Page object to analyze.
        page_num: Zero-indexed page number.

    Returns:
        PageAnalysis for this page.
    """
    # Extract text
    text = page.get_text().strip()
    char_count = len(text)
    has_text = char_count > 0

    # Determine if page is image-only
    # A page is image-only if it has images but no text blocks
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = text_dict.get("blocks", [])

    # Block type 0 = text, type 1 = image
    has_text_blocks = any(block.get("type") == 0 for block in blocks)
    has_image_blocks = any(block.get("type") == 1 for block in blocks)

    # Page is image-only if:
    # - It has image blocks AND
    # - It has no text blocks (or very little text)
    is_image_only = has_image_blocks and not has_text_blocks

    return PageAnalysis(
        page_number=page_num,
        has_text=has_text,
        text_char_count=char_count,
        is_image_only=is_image_only,
    )
