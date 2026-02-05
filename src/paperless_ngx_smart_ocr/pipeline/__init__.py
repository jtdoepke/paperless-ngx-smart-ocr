"""Processing pipeline module for OCR and Markdown conversion.

This module provides the document processing pipeline for paperless-ngx-smart-ocr,
including:

- **Stage 1 (OCR)**: Add searchable text layers to scanned PDFs using OCRmyPDF
  with optional Surya layout detection for complex documents.
- **Stage 2 (Markdown)**: Convert PDFs to structured Markdown using Marker,
  optionally with LLM assistance for complex elements.

Example:
    ```python
    from paperless_ngx_smart_ocr.config import get_settings
    from paperless_ngx_smart_ocr.pipeline import (
        Stage1Processor,
        Stage2Processor,
        analyze_document,
        process_stage1,
        process_stage2,
    )

    # Analyze a document
    analysis = analyze_document(pdf_path)
    if analysis.is_born_digital:
        print("Document already has text")

    # Process through Stage 1 (OCR)
    settings = get_settings()
    result1 = await process_stage1(
        input_path,
        output_path,
        config=settings.pipeline.stage1,
    )

    # Process through Stage 2 (Markdown)
    result2 = await process_stage2(
        output_path,
        config=settings.pipeline.stage2,
    )
    if result2.success:
        print(result2.markdown[:500])
    ```
"""

from __future__ import annotations

from paperless_ngx_smart_ocr.pipeline.exceptions import (
    LayoutDetectionError,
    MarkerConversionError,
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
    Stage2Result,
)
from paperless_ngx_smart_ocr.pipeline.preprocessing import (
    analyze_document,
    has_text_layer,
)
from paperless_ngx_smart_ocr.pipeline.stage1_ocr import Stage1Processor, process_stage1
from paperless_ngx_smart_ocr.pipeline.stage2_markdown import (
    Stage2Processor,
    clear_marker_models,
    get_marker_models,
    postprocess_markdown,
    process_stage2,
)


__all__ = [
    "BoundingBox",
    "DocumentAnalysis",
    "LayoutDetectionError",
    "LayoutDetector",
    "LayoutRegion",
    "LayoutResult",
    "MarkerConversionError",
    "OCRError",
    "PageAnalysis",
    "PipelineError",
    "PreprocessingError",
    "RegionLabel",
    "Stage1Processor",
    "Stage1Result",
    "Stage2Processor",
    "Stage2Result",
    "analyze_document",
    "clear_marker_models",
    "clear_predictor_cache",
    "get_layout_predictor",
    "get_marker_models",
    "has_text_layer",
    "postprocess_markdown",
    "process_stage1",
    "process_stage2",
]
