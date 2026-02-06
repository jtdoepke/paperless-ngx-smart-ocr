"""Data models for the processing pipeline.

This module defines dataclasses and enums used throughout the pipeline
for representing layout detection results, document analysis, and
processing outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "BoundingBox",
    "DocumentAnalysis",
    "LayoutRegion",
    "LayoutResult",
    "PageAnalysis",
    "PipelineResult",
    "RegionLabel",
    "Stage1Result",
    "Stage2Result",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RegionLabel(StrEnum):
    """Layout region labels detected by Surya.

    These labels categorize different types of content regions in a document,
    following Surya's layout detection model output.

    Attributes:
        CAPTION: Image or figure caption.
        FOOTNOTE: Footnote text.
        FORMULA: Mathematical formula.
        LIST_ITEM: Item in a list.
        PAGE_FOOTER: Page footer content.
        PAGE_HEADER: Page header content.
        PICTURE: Image or photograph.
        FIGURE: Figure or diagram.
        SECTION_HEADER: Section or chapter heading.
        TABLE: Tabular data.
        FORM: Form elements.
        TABLE_OF_CONTENTS: Table of contents entries.
        HANDWRITING: Handwritten text.
        TEXT: Regular body text.
        TEXT_INLINE_MATH: Text containing inline math.
    """

    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "List-item"
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    FIGURE = "Figure"
    SECTION_HEADER = "Section-header"
    TABLE = "Table"
    FORM = "Form"
    TABLE_OF_CONTENTS = "Table-of-contents"
    HANDWRITING = "Handwriting"
    TEXT = "Text"
    TEXT_INLINE_MATH = "Text-inline-math"


# ---------------------------------------------------------------------------
# Layout Detection Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box with coordinates (x0, y0, x1, y1).

    Coordinates are in pixels, with origin at top-left corner.

    Attributes:
        x0: Left edge x-coordinate.
        y0: Top edge y-coordinate.
        x1: Right edge x-coordinate.
        y1: Bottom edge y-coordinate.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to a tuple (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        """Convert to integer tuple for PIL cropping."""
        return (int(self.x0), int(self.y0), int(self.x1), int(self.y1))


@dataclass(frozen=True, slots=True)
class LayoutRegion:
    """A detected layout region from Surya.

    Represents a single content region detected in a document page,
    with its bounding box, type classification, and reading order position.

    Attributes:
        bbox: Bounding box coordinates for the region.
        label: Classification of the region type.
        position: Reading order position (0 = first in reading order).
        confidence: Detection confidence score (0.0 to 1.0).
        polygon: Optional polygon vertices for non-rectangular regions.
    """

    bbox: BoundingBox
    label: RegionLabel
    position: int
    confidence: float
    polygon: tuple[tuple[float, float], ...] | None = None

    def should_ocr(self, exclude_labels: set[str]) -> bool:
        """Determine if this region should be included in OCR.

        Args:
            exclude_labels: Set of label names to exclude (case-insensitive,
                supports both underscore and hyphen separators).

        Returns:
            True if the region should be OCR'd, False if excluded.
        """
        # Normalize the region's label for comparison
        label_normalized = self.label.value.lower().replace("-", "_")

        for exclude in exclude_labels:
            exclude_normalized = exclude.lower().replace("-", "_")
            if exclude_normalized == label_normalized:
                return False

        return True


@dataclass(slots=True)
class LayoutResult:
    """Layout detection results for a single page.

    Contains all detected regions for a page along with image dimensions.

    Attributes:
        page_number: Zero-indexed page number.
        regions: List of detected layout regions.
        image_bbox: Bounding box of the full page image.
        image_width: Width of the page image in pixels.
        image_height: Height of the page image in pixels.
    """

    page_number: int
    regions: list[LayoutRegion]
    image_bbox: BoundingBox
    image_width: int
    image_height: int

    def get_text_regions(
        self,
        *,
        exclude_labels: set[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[LayoutRegion]:
        """Get text regions sorted by reading order.

        Args:
            exclude_labels: Region labels to exclude from results.
            min_confidence: Minimum confidence threshold for inclusion.

        Returns:
            List of regions sorted by reading order position.
        """
        exclude = exclude_labels or set()
        regions = [
            r
            for r in self.regions
            if r.should_ocr(exclude) and r.confidence >= min_confidence
        ]
        return sorted(regions, key=lambda r: r.position)

    def get_regions_by_label(self, label: RegionLabel) -> list[LayoutRegion]:
        """Get all regions of a specific type.

        Args:
            label: The region label to filter by.

        Returns:
            List of regions with the specified label, sorted by position.
        """
        regions = [r for r in self.regions if r.label == label]
        return sorted(regions, key=lambda r: r.position)


# ---------------------------------------------------------------------------
# Document Analysis Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PageAnalysis:
    """Analysis results for a single page.

    Contains information about text content and page characteristics.

    Attributes:
        page_number: Zero-indexed page number.
        has_text: Whether the page has extractable text content.
        text_char_count: Number of text characters on the page.
        is_image_only: Whether the page contains only raster images.
    """

    page_number: int
    has_text: bool
    text_char_count: int
    is_image_only: bool


@dataclass(slots=True)
class DocumentAnalysis:
    """Analysis results for an entire document.

    Aggregates page-level analysis into document-level statistics.

    Attributes:
        total_pages: Total number of pages in the document.
        pages: Per-page analysis results.
        has_any_text: Whether any page has extractable text.
        is_born_digital: Whether the document appears to be born-digital
            (created from digital sources, not scanned).
    """

    total_pages: int
    pages: list[PageAnalysis]
    has_any_text: bool
    is_born_digital: bool

    @property
    def pages_with_text(self) -> list[PageAnalysis]:
        """Get pages that have extractable text."""
        return [p for p in self.pages if p.has_text]

    @property
    def pages_without_text(self) -> list[PageAnalysis]:
        """Get pages without extractable text (likely scanned)."""
        return [p for p in self.pages if not p.has_text]

    @property
    def total_text_chars(self) -> int:
        """Total character count across all pages."""
        return sum(p.text_char_count for p in self.pages)


# ---------------------------------------------------------------------------
# Processing Result Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Stage1Result:
    """Result of Stage 1 OCR processing.

    Captures the outcome of processing a document through the OCR pipeline,
    including success/failure status, timing, and any generated outputs.

    Attributes:
        success: Whether processing completed successfully.
        input_path: Path to the input PDF file.
        output_path: Path to the OCR'd output PDF, if created.
        document_analysis: Analysis of the input document.
        layout_results: Layout detection results per page, if performed.
        pages_processed: Number of pages that were OCR'd.
        skipped: Whether processing was skipped (e.g., born-digital).
        skip_reason: Human-readable reason for skipping, if applicable.
        error: Error message if processing failed.
        processing_time_seconds: Total processing time in seconds.
        created_at: Timestamp when processing started.
    """

    success: bool
    input_path: Path
    output_path: Path | None
    document_analysis: DocumentAnalysis
    layout_results: list[LayoutResult] | None
    pages_processed: int
    skipped: bool
    skip_reason: str | None
    error: str | None = None
    processing_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        """Whether this result represents a terminal state."""
        return self.success or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "success": self.success,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "pages_processed": self.pages_processed,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat(),
            "document_analysis": {
                "total_pages": self.document_analysis.total_pages,
                "has_any_text": self.document_analysis.has_any_text,
                "is_born_digital": self.document_analysis.is_born_digital,
                "total_text_chars": self.document_analysis.total_text_chars,
            },
            "layout_results_count": (
                len(self.layout_results) if self.layout_results else 0
            ),
        }


@dataclass(slots=True)
class Stage2Result:
    """Result of Stage 2 Markdown conversion.

    Captures the outcome of processing a document through the Markdown
    conversion pipeline, including success/failure status, timing, and
    the extracted Markdown content.

    Attributes:
        success: Whether processing completed successfully.
        input_path: Path to the input PDF file.
        markdown: Extracted Markdown content.
        page_count: Number of pages processed.
        images: Extracted images as block_id -> base64 string mapping.
        metadata: Marker metadata (table of contents, page stats).
        llm_used: Whether LLM assistance was used for conversion.
        skipped: Whether processing was skipped.
        skip_reason: Human-readable reason for skipping, if applicable.
        error: Error message if processing failed.
        processing_time_seconds: Total processing time in seconds.
        created_at: Timestamp when processing started.
    """

    success: bool
    input_path: Path
    markdown: str
    page_count: int
    images: dict[str, str]
    metadata: dict[str, Any]
    llm_used: bool
    skipped: bool
    skip_reason: str | None
    error: str | None = None
    processing_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        """Whether this result represents a terminal state."""
        return self.success or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "success": self.success,
            "input_path": str(self.input_path),
            "markdown_length": len(self.markdown),
            "page_count": self.page_count,
            "images_count": len(self.images),
            "llm_used": self.llm_used,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Pipeline Result Model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """Result of the full pipeline orchestration.

    Captures the outcome of processing a document through both Stage 1 (OCR)
    and Stage 2 (Markdown), including tag and content update status.

    Attributes:
        document_id: The paperless-ngx document ID.
        success: Whether the overall pipeline completed successfully.
        stage1_result: Stage 1 OCR result, or None if not run.
        stage2_result: Stage 2 Markdown result, or None if not run.
        stage1_skipped_by_config: Whether Stage 1 was skipped due to config.
        stage2_skipped_by_config: Whether Stage 2 was skipped due to config.
        tags_updated: Whether tags were successfully updated.
        content_updated: Whether document content was successfully updated.
        document_uploaded: Whether the OCR'd PDF was re-uploaded.
        error: Error message if the pipeline failed.
        processing_time_seconds: Total pipeline processing time in seconds.
        created_at: Timestamp when processing started.
    """

    document_id: int
    success: bool
    stage1_result: Stage1Result | None
    stage2_result: Stage2Result | None
    stage1_skipped_by_config: bool
    stage2_skipped_by_config: bool
    tags_updated: bool
    content_updated: bool
    document_uploaded: bool
    dry_run: bool = False
    error: str | None = None
    processing_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        """Whether this result represents a terminal state."""
        return self.success or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "document_id": self.document_id,
            "success": self.success,
            "dry_run": self.dry_run,
            "stage1_result": (
                self.stage1_result.to_dict() if self.stage1_result else None
            ),
            "stage2_result": (
                self.stage2_result.to_dict() if self.stage2_result else None
            ),
            "stage1_skipped_by_config": self.stage1_skipped_by_config,
            "stage2_skipped_by_config": self.stage2_skipped_by_config,
            "tags_updated": self.tags_updated,
            "content_updated": self.content_updated,
            "document_uploaded": self.document_uploaded,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat(),
        }
