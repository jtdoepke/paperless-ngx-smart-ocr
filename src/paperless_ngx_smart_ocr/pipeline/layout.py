"""Surya layout detection wrapper.

This module provides layout detection using Surya's LayoutPredictor,
with model caching and result conversion to pipeline models.

Layout detection identifies text regions, tables, figures, and other
elements in document pages, along with their reading order. This
information can be used to improve OCR accuracy and document structure
understanding.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import fitz  # PyMuPDF
from PIL import Image

from paperless_ngx_smart_ocr.config import GPUMode
from paperless_ngx_smart_ocr.observability import get_logger
from paperless_ngx_smart_ocr.pipeline.exceptions import LayoutDetectionError
from paperless_ngx_smart_ocr.pipeline.models import (
    BoundingBox,
    LayoutRegion,
    LayoutResult,
    RegionLabel,
)


if TYPE_CHECKING:
    from pathlib import Path

    from surya.layout import LayoutPredictor as SuryaLayoutPredictor


__all__ = [
    "LayoutDetector",
    "clear_predictor_cache",
    "get_layout_predictor",
]


# Module-level cache for the predictor
_predictor_cache: dict[GPUMode, Any] = {}


def get_layout_predictor(gpu_mode: GPUMode = GPUMode.AUTO) -> SuryaLayoutPredictor:
    """Get a cached Surya LayoutPredictor instance.

    The predictor is cached to avoid reloading the model on each call.
    Models are automatically downloaded on first use.

    Args:
        gpu_mode: GPU mode for the predictor. AUTO will use GPU if available.

    Returns:
        Cached LayoutPredictor instance.

    Raises:
        LayoutDetectionError: If the predictor cannot be initialized.

    Note:
        The cache is global to the process. Call clear_predictor_cache()
        to release memory if needed.
    """
    if gpu_mode in _predictor_cache:
        return _predictor_cache[gpu_mode]

    logger = get_logger(__name__)
    logger.info("loading_layout_predictor", gpu_mode=gpu_mode.value)

    try:
        # Import Surya here to avoid loading models at module import time
        from surya.layout import LayoutPredictor

        # Configure device based on gpu_mode
        # Surya uses device auto-detection by default
        # For explicit control, environment variables can be set before import
        device = None
        if gpu_mode == GPUMode.CPU:
            device = "cpu"
        elif gpu_mode == GPUMode.CUDA:
            device = "cuda"
        # AUTO lets Surya decide

        predictor = LayoutPredictor(device=device)
        _predictor_cache[gpu_mode] = predictor

        logger.info("layout_predictor_loaded", gpu_mode=gpu_mode.value)

    except ImportError as exc:
        msg = "Surya is not installed. Install with: pip install surya-ocr"
        raise LayoutDetectionError(msg, cause=exc) from exc
    except Exception as exc:
        msg = f"Failed to initialize layout predictor: {exc}"
        raise LayoutDetectionError(msg, cause=exc) from exc
    else:
        return predictor


def clear_predictor_cache() -> None:
    """Clear the cached predictor to free memory.

    Call this if you need to release GPU memory or reload the model.
    """
    _predictor_cache.clear()


class LayoutDetector:
    """Wrapper for Surya layout detection with PDF support.

    Handles conversion from PDF pages to images and processes layout
    detection results into pipeline models. The predictor is lazily
    loaded and cached for efficiency.

    Attributes:
        DEFAULT_DPI: Default DPI for rendering PDF pages (150).

    Example:
        ```python
        detector = LayoutDetector(gpu_mode=GPUMode.AUTO)
        results = detector.detect_layout(pdf_path)
        for page_result in results:
            for region in page_result.get_text_regions():
                print(f"Region {region.position}: {region.label}")
        ```
    """

    DEFAULT_DPI = 150  # Balance between quality and speed

    def __init__(
        self,
        *,
        gpu_mode: GPUMode = GPUMode.AUTO,
        dpi: int | None = None,
    ) -> None:
        """Initialize the layout detector.

        Args:
            gpu_mode: GPU acceleration mode for the predictor.
            dpi: DPI for rendering PDF pages to images. Higher values
                improve accuracy but use more memory. Default is 150.
        """
        self._gpu_mode = gpu_mode
        self._dpi = dpi or self.DEFAULT_DPI
        self._predictor: SuryaLayoutPredictor | None = None
        self._logger = get_logger(__name__)

    @property
    def predictor(self) -> SuryaLayoutPredictor:
        """Get the layout predictor, loading if necessary."""
        if self._predictor is None:
            self._predictor = get_layout_predictor(self._gpu_mode)
        return self._predictor

    def detect_layout(
        self,
        pdf_path: Path,
        *,
        page_numbers: list[int] | None = None,
    ) -> list[LayoutResult]:
        """Detect layout for all or specified pages of a PDF.

        Renders each page to an image and runs Surya layout detection
        to identify text regions, tables, figures, and their reading order.

        Args:
            pdf_path: Path to the PDF file to analyze.
            page_numbers: Specific page numbers to process (0-indexed).
                If None, all pages are processed.

        Returns:
            List of LayoutResult, one per processed page.

        Raises:
            LayoutDetectionError: If layout detection fails.

        Example:
            ```python
            # Detect layout for all pages
            results = detector.detect_layout(pdf_path)

            # Detect layout for specific pages
            results = detector.detect_layout(pdf_path, page_numbers=[0, 1, 2])
            ```
        """
        self._logger.debug(
            "detecting_layout",
            path=str(pdf_path),
            page_numbers=page_numbers,
            dpi=self._dpi,
        )

        try:
            with fitz.open(pdf_path) as doc:
                pages_to_process = (
                    page_numbers if page_numbers is not None else list(range(len(doc)))
                )

                # Validate page numbers
                valid_pages = [p for p in pages_to_process if 0 <= p < len(doc)]
                if not valid_pages:
                    self._logger.warning(
                        "no_valid_pages",
                        path=str(pdf_path),
                        requested=pages_to_process,
                        total_pages=len(doc),
                    )
                    return []

                # Render pages to images
                images: list[Image.Image] = []
                page_info: list[tuple[int, int, int]] = []  # (page_num, width, height)

                for page_num in valid_pages:
                    page = doc[page_num]
                    # Render at specified DPI
                    zoom = self._dpi / 72  # 72 is PDF default DPI
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    images.append(img)
                    page_info.append((page_num, pix.width, pix.height))

                if not images:
                    return []

                # Run layout detection
                predictions = self.predictor(images)

                # Convert to pipeline models
                results: list[LayoutResult] = []
                for (page_num, width, height), pred in zip(
                    page_info, predictions, strict=True
                ):
                    result = self._convert_prediction(pred, page_num, width, height)
                    results.append(result)

                self._logger.info(
                    "layout_detected",
                    path=str(pdf_path),
                    pages_processed=len(results),
                    total_regions=sum(len(r.regions) for r in results),
                )

                return results

        except LayoutDetectionError:
            raise
        except fitz.FileDataError as exc:
            msg = f"Invalid or corrupted PDF file: {pdf_path}"
            raise LayoutDetectionError(msg, cause=exc) from exc
        except Exception as exc:
            msg = f"Layout detection failed: {exc}"
            raise LayoutDetectionError(msg, cause=exc) from exc

    def detect_single_page(
        self,
        image: Image.Image,
        page_number: int = 0,
    ) -> LayoutResult:
        """Detect layout for a single PIL Image.

        Useful when you already have a rendered page image and don't
        need to process a full PDF.

        Args:
            image: PIL Image to analyze.
            page_number: Page number to assign to the result (for tracking).

        Returns:
            LayoutResult for the image.

        Raises:
            LayoutDetectionError: If detection fails.
        """
        self._logger.debug(
            "detecting_single_page",
            page_number=page_number,
            image_size=(image.width, image.height),
        )

        try:
            predictions = self.predictor([image])
            return self._convert_prediction(
                predictions[0],
                page_number,
                image.width,
                image.height,
            )
        except Exception as exc:
            msg = f"Layout detection failed for page {page_number}: {exc}"
            raise LayoutDetectionError(msg, page=page_number, cause=exc) from exc

    def _convert_prediction(
        self,
        prediction: Any,  # noqa: ANN401
        page_number: int,
        width: int,
        height: int,
    ) -> LayoutResult:
        """Convert Surya prediction to LayoutResult.

        Handles the conversion of Surya's output format to our pipeline
        models, including label normalization and confidence extraction.

        Args:
            prediction: Raw prediction from Surya (LayoutResult object).
            page_number: The page number (0-indexed).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Converted LayoutResult.
        """
        regions: list[LayoutRegion] = []

        # Surya returns a LayoutResult object with bboxes attribute
        bboxes = getattr(prediction, "bboxes", [])

        for idx, bbox_data in enumerate(bboxes):
            # Extract bounding box coordinates
            # Surya bbox objects have bbox attribute as list [x0, y0, x1, y1]
            bbox_coords = getattr(bbox_data, "bbox", [0, 0, 0, 0])
            bbox = BoundingBox(
                x0=float(bbox_coords[0]),
                y0=float(bbox_coords[1]),
                x1=float(bbox_coords[2]),
                y1=float(bbox_coords[3]),
            )

            # Extract label
            label_str = getattr(bbox_data, "label", "Text")
            try:
                label = RegionLabel(label_str)
            except ValueError:
                # Unknown label, default to TEXT
                self._logger.debug(
                    "unknown_region_label",
                    label=label_str,
                    page=page_number,
                )
                label = RegionLabel.TEXT

            # Extract confidence
            # Surya provides confidence directly or via top_k
            confidence = getattr(bbox_data, "confidence", 0.0)
            if confidence == 0.0:
                top_k = getattr(bbox_data, "top_k", {})
                if isinstance(top_k, dict):
                    confidence = top_k.get(label_str, 0.5)
                else:
                    confidence = 0.5

            # Extract position (reading order)
            position = getattr(bbox_data, "position", idx)

            # Extract polygon if present
            polygon_data = getattr(bbox_data, "polygon", None)
            polygon: tuple[tuple[float, float], ...] | None = None
            if polygon_data:
                with contextlib.suppress(TypeError, ValueError):
                    polygon = tuple(tuple(pt) for pt in polygon_data)

            regions.append(
                LayoutRegion(
                    bbox=bbox,
                    label=label,
                    position=position,
                    confidence=float(confidence),
                    polygon=polygon,
                )
            )

        # Create image bounding box
        image_bbox = BoundingBox(x0=0, y0=0, x1=float(width), y1=float(height))

        return LayoutResult(
            page_number=page_number,
            regions=regions,
            image_bbox=image_bbox,
            image_width=width,
            image_height=height,
        )
