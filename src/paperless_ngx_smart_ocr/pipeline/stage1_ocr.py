"""Stage 1: OCR Pipeline.

This module implements the OCR processing pipeline that:
1. Analyzes documents for existing text (born-digital detection)
2. Optionally runs Surya layout detection
3. Processes with OCRmyPDF to add searchable text layers

The pipeline respects configuration for born-digital handling and
layout detection settings.
"""

from __future__ import annotations

import shutil
import time
from typing import TYPE_CHECKING

import ocrmypdf

from paperless_ngx_smart_ocr.config import BornDigitalHandling, GPUMode, Stage1Config
from paperless_ngx_smart_ocr.observability import get_logger
from paperless_ngx_smart_ocr.pipeline.exceptions import (
    LayoutDetectionError,
    OCRError,
    PipelineError,
)
from paperless_ngx_smart_ocr.pipeline.layout import LayoutDetector
from paperless_ngx_smart_ocr.pipeline.models import (
    DocumentAnalysis,
    LayoutResult,
    Stage1Result,
)
from paperless_ngx_smart_ocr.pipeline.preprocessing import analyze_document


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "Stage1Processor",
    "process_stage1",
]


class Stage1Processor:
    """OCR processor for Stage 1 of the pipeline.

    Handles the complete OCR workflow:
    1. Document analysis (born-digital detection)
    2. Layout detection (optional, using Surya)
    3. OCR processing (using OCRmyPDF)

    The processor can be configured to skip documents that already have
    text layers (born-digital) or force OCR on all documents.

    Attributes:
        config: Stage 1 configuration settings.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings

        settings = get_settings()
        processor = Stage1Processor(
            config=settings.pipeline.stage1,
            gpu_mode=settings.gpu.enabled,
        )

        result = await processor.process(input_path, output_path)
        if result.success:
            print(f"OCR completed: {result.pages_processed} pages")
        elif result.skipped:
            print(f"Skipped: {result.skip_reason}")
        else:
            print(f"Failed: {result.error}")
        ```
    """

    def __init__(
        self,
        *,
        config: Stage1Config,
        gpu_mode: GPUMode = GPUMode.AUTO,
    ) -> None:
        """Initialize the Stage 1 processor.

        Args:
            config: Stage 1 configuration from settings.
            gpu_mode: GPU mode for layout detection.
        """
        self._config = config
        self._gpu_mode = gpu_mode
        self._logger = get_logger(__name__)

        # Initialize layout detector if enabled
        self._layout_detector: LayoutDetector | None = None
        if config.layout_detection.enabled:
            self._layout_detector = LayoutDetector(gpu_mode=gpu_mode)

    @property
    def config(self) -> Stage1Config:
        """Get the Stage 1 configuration."""
        return self._config

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        force: bool = False,
    ) -> Stage1Result:
        """Process a document through Stage 1 OCR.

        Performs document analysis, optional layout detection, and OCR
        processing. The workflow respects born-digital handling settings
        unless force=True is specified.

        Args:
            input_path: Path to input PDF file.
            output_path: Path for output PDF with OCR layer.
            force: If True, ignore born-digital handling and always OCR.

        Returns:
            Stage1Result with processing outcome, timing, and any errors.

        Raises:
            PipelineError: If processing fails with an unrecoverable error.

        Note:
            This method is async to integrate with the job queue, but the
            underlying OCR operations are synchronous.
        """
        start_time = time.monotonic()
        self._logger.info(
            "stage1_starting",
            input=str(input_path),
            output=str(output_path),
            force=force,
        )

        # Initialize variables for result
        analysis: DocumentAnalysis | None = None
        layout_results: list[LayoutResult] | None = None

        try:
            # Step 1: Analyze document
            analysis = analyze_document(input_path)

            # Step 2: Check born-digital handling
            if not force and self._should_skip_ocr(analysis):
                processing_time = time.monotonic() - start_time
                self._logger.info(
                    "stage1_skipped",
                    input=str(input_path),
                    reason="born-digital",
                    processing_time_seconds=processing_time,
                )
                return Stage1Result(
                    success=True,
                    input_path=input_path,
                    output_path=None,
                    document_analysis=analysis,
                    layout_results=None,
                    pages_processed=0,
                    skipped=True,
                    skip_reason="Document has existing text layer (born-digital)",
                    processing_time_seconds=processing_time,
                )

            # Step 3: Layout detection (if enabled)
            if self._layout_detector is not None:
                layout_results = self._detect_layout_safe(input_path)

            # Step 4: Run OCRmyPDF
            self._run_ocrmypdf(input_path, output_path)

            processing_time = time.monotonic() - start_time
            self._logger.info(
                "stage1_completed",
                input=str(input_path),
                output=str(output_path),
                pages_processed=analysis.total_pages,
                processing_time_seconds=processing_time,
            )

            return Stage1Result(
                success=True,
                input_path=input_path,
                output_path=output_path,
                document_analysis=analysis,
                layout_results=layout_results,
                pages_processed=analysis.total_pages,
                skipped=False,
                skip_reason=None,
                processing_time_seconds=processing_time,
            )

        except OCRError as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage1_ocr_failed",
                input=str(input_path),
                error=str(exc),
            )
            # Return a failed result rather than raising
            return Stage1Result(
                success=False,
                input_path=input_path,
                output_path=None,
                document_analysis=analysis or self._empty_analysis(),
                layout_results=layout_results,
                pages_processed=0,
                skipped=False,
                skip_reason=None,
                error=str(exc),
                processing_time_seconds=processing_time,
            )

        except PipelineError as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage1_failed",
                input=str(input_path),
                error=str(exc),
            )
            # Return a failed result
            return Stage1Result(
                success=False,
                input_path=input_path,
                output_path=None,
                document_analysis=analysis or self._empty_analysis(),
                layout_results=layout_results,
                pages_processed=0,
                skipped=False,
                skip_reason=None,
                error=str(exc),
                processing_time_seconds=processing_time,
            )

        except Exception as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage1_unexpected_error",
                input=str(input_path),
                error=str(exc),
            )
            # Re-raise unexpected exceptions as PipelineError
            msg = f"Stage 1 processing failed unexpectedly: {exc}"
            raise PipelineError(msg) from exc

    def _should_skip_ocr(self, analysis: DocumentAnalysis) -> bool:
        """Determine if OCR should be skipped based on analysis.

        Args:
            analysis: Document analysis results.

        Returns:
            True if OCR should be skipped, False otherwise.
        """
        if self._config.born_digital_handling == BornDigitalHandling.FORCE:
            return False

        # SKIP mode: skip if document has substantial text (is born-digital)
        if analysis.is_born_digital:
            self._logger.debug(
                "skipping_born_digital",
                reason="Document appears to be born-digital",
                pages_with_text=len(analysis.pages_with_text),
                total_pages=analysis.total_pages,
            )
            return True

        return False

    def _detect_layout_safe(self, pdf_path: Path) -> list[LayoutResult] | None:
        """Run layout detection with graceful error handling.

        If layout detection fails, logs a warning and returns None
        rather than failing the entire pipeline.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Layout detection results, or None if detection fails.
        """
        if self._layout_detector is None:
            return None

        self._logger.debug("running_layout_detection", path=str(pdf_path))

        try:
            return self._layout_detector.detect_layout(pdf_path)
        except LayoutDetectionError as exc:
            # Log warning but continue without layout detection
            self._logger.warning(
                "layout_detection_failed",
                path=str(pdf_path),
                error=str(exc),
            )
            return None

    def _run_ocrmypdf(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:
        """Run OCRmyPDF on the document.

        Args:
            input_path: Input PDF path.
            output_path: Output PDF path.

        Raises:
            OCRError: If OCRmyPDF fails.
        """
        ocr_config = self._config.ocrmypdf
        self._logger.debug(
            "running_ocrmypdf",
            input=str(input_path),
            output=str(output_path),
            language=ocr_config.language,
            deskew=ocr_config.deskew,
            clean=ocr_config.clean,
            rotate_pages=ocr_config.rotate_pages,
        )

        try:
            # Determine skip_text and force_ocr based on born_digital_handling
            skip_text = self._config.born_digital_handling == BornDigitalHandling.SKIP
            force_ocr = self._config.born_digital_handling == BornDigitalHandling.FORCE

            # Run OCRmyPDF
            # Note: ocrmypdf.ocr returns an ExitCode enum
            exit_code = ocrmypdf.ocr(
                str(input_path),
                str(output_path),
                deskew=ocr_config.deskew,
                clean=ocr_config.clean,
                rotate_pages=ocr_config.rotate_pages,
                language=[ocr_config.language],
                skip_text=skip_text,
                force_ocr=force_ocr,
            )

            # OCRmyPDF returns ExitCode enum, 0 is success
            if exit_code != 0:
                msg = f"OCRmyPDF exited with code {exit_code}"
                raise OCRError(msg, exit_code=int(exit_code))  # noqa: TRY301

            self._logger.debug(
                "ocrmypdf_completed",
                input=str(input_path),
                output=str(output_path),
            )

        except ocrmypdf.exceptions.PriorOcrFoundError:
            # Document already has OCR - copy input to output
            self._logger.info(
                "ocrmypdf_prior_ocr_found",
                input=str(input_path),
            )
            shutil.copy2(input_path, output_path)

        except ocrmypdf.exceptions.MissingDependencyError as exc:
            msg = f"Missing OCRmyPDF dependency: {exc}"
            raise OCRError(msg, cause=exc) from exc

        except ocrmypdf.exceptions.InputFileError as exc:
            msg = f"Invalid input file: {exc}"
            raise OCRError(msg, cause=exc) from exc

        except ocrmypdf.exceptions.EncryptedPdfError as exc:
            msg = "PDF is encrypted and cannot be processed"
            raise OCRError(msg, cause=exc) from exc

        except OCRError:
            raise

        except Exception as exc:
            msg = f"OCRmyPDF failed: {exc}"
            raise OCRError(msg, cause=exc) from exc

    def _empty_analysis(self) -> DocumentAnalysis:
        """Create an empty DocumentAnalysis for error cases.

        Returns:
            DocumentAnalysis with zero pages and no text.
        """
        return DocumentAnalysis(
            total_pages=0,
            pages=[],
            has_any_text=False,
            is_born_digital=False,
        )


async def process_stage1(
    input_path: Path,
    output_path: Path,
    *,
    config: Stage1Config,
    gpu_mode: GPUMode = GPUMode.AUTO,
    force: bool = False,
) -> Stage1Result:
    """Convenience function to process a document through Stage 1.

    Creates a Stage1Processor and processes the document in a single call.

    Args:
        input_path: Path to input PDF file.
        output_path: Path for output PDF with OCR layer.
        config: Stage 1 configuration from settings.
        gpu_mode: GPU mode for layout detection.
        force: If True, force OCR regardless of born-digital status.

    Returns:
        Stage1Result with processing outcome.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings

        settings = get_settings()
        result = await process_stage1(
            input_path,
            output_path,
            config=settings.pipeline.stage1,
            gpu_mode=settings.gpu.enabled,
        )
        ```
    """
    processor = Stage1Processor(config=config, gpu_mode=gpu_mode)
    return await processor.process(input_path, output_path, force=force)
