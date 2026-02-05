"""Processing pipeline module for OCR and Markdown conversion.

This module provides the document processing pipeline for paperless-ngx-smart-ocr,
including:

- **Stage 1 (OCR)**: Add searchable text layers to scanned PDFs using OCRmyPDF
  with optional Surya layout detection for complex documents.
- **Stage 2 (Markdown)**: Convert PDFs to structured Markdown using Marker
  (implemented separately).

Example:
    ```python
    from paperless_ngx_smart_ocr.config import get_settings
    from paperless_ngx_smart_ocr.pipeline import (
        Stage1Processor,
        analyze_document,
        process_stage1,
    )

    # Analyze a document
    analysis = analyze_document(pdf_path)
    if analysis.is_born_digital:
        print("Document already has text")

    # Process through Stage 1
    settings = get_settings()
    result = await process_stage1(
        input_path,
        output_path,
        config=settings.pipeline.stage1,
    )
    ```
"""

from __future__ import annotations

from paperless_ngx_smart_ocr.pipeline.exceptions import (
    LayoutDetectionError,
    OCRError,
    PipelineError,
    PreprocessingError,
)
from paperless_ngx_smart_ocr.pipeline.layout import (
    LayoutDetector,
    clear_predictor_cache,
    get_layout_predictor,
)
from paperless_ngx_smart_ocr.pipeline.models import (
    BoundingBox,
    DocumentAnalysis,
    LayoutRegion,
    LayoutResult,
    PageAnalysis,
    RegionLabel,
    Stage1Result,
)
from paperless_ngx_smart_ocr.pipeline.preprocessing import (
    analyze_document,
    has_text_layer,
)
from paperless_ngx_smart_ocr.pipeline.stage1_ocr import Stage1Processor, process_stage1


__all__ = [
    "BoundingBox",
    "DocumentAnalysis",
    "LayoutDetectionError",
    "LayoutDetector",
    "LayoutRegion",
    "LayoutResult",
    "OCRError",
    "PageAnalysis",
    "PipelineError",
    "PreprocessingError",
    "RegionLabel",
    "Stage1Processor",
    "Stage1Result",
    "analyze_document",
    "clear_predictor_cache",
    "get_layout_predictor",
    "has_text_layer",
    "process_stage1",
]
