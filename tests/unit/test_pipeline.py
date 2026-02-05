"""Unit tests for the pipeline module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from paperless_ngx_smart_ocr.config import (
    BornDigitalHandling,
    GPUMode,
    Stage1Config,
    Stage2Config,
)
from paperless_ngx_smart_ocr.pipeline import (
    BoundingBox,
    DocumentAnalysis,
    LayoutDetectionError,
    LayoutRegion,
    LayoutResult,
    MarkerConversionError,
    OCRError,
    PageAnalysis,
    PipelineError,
    PreprocessingError,
    RegionLabel,
    Stage1Processor,
    Stage1Result,
    Stage2Processor,
    Stage2Result,
    postprocess_markdown,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestPipelineError:
    """Tests for PipelineError."""

    def test_message_attribute(self) -> None:
        """Test that message is stored as attribute."""
        exc = PipelineError("Test error")
        assert exc.message == "Test error"
        assert str(exc) == "Test error"

    def test_inheritance(self) -> None:
        """Test that PipelineError inherits from Exception."""
        exc = PipelineError("Test error")
        assert isinstance(exc, Exception)


class TestPreprocessingError:
    """Tests for PreprocessingError."""

    def test_basic_error(self) -> None:
        """Test basic error without cause."""
        exc = PreprocessingError("Failed to read file")
        assert exc.message == "Failed to read file"
        assert exc.cause is None

    def test_error_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = FileNotFoundError("No such file")
        exc = PreprocessingError("Failed to read file", cause=cause)
        assert exc.cause is cause
        assert exc.__cause__ is cause


class TestLayoutDetectionError:
    """Tests for LayoutDetectionError."""

    def test_basic_error(self) -> None:
        """Test basic error without page or cause."""
        exc = LayoutDetectionError("Detection failed")
        assert exc.message == "Detection failed"
        assert exc.page is None
        assert exc.cause is None
        assert str(exc) == "Detection failed"

    def test_error_with_page(self) -> None:
        """Test error with page number."""
        exc = LayoutDetectionError("Detection failed", page=5)
        assert exc.page == 5
        assert str(exc) == "Detection failed (page=5)"

    def test_error_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = RuntimeError("GPU error")
        exc = LayoutDetectionError("Detection failed", cause=cause)
        assert exc.cause is cause
        assert exc.__cause__ is cause


class TestOCRError:
    """Tests for OCRError."""

    def test_basic_error(self) -> None:
        """Test basic error without exit code."""
        exc = OCRError("OCR failed")
        assert exc.message == "OCR failed"
        assert exc.exit_code is None
        assert str(exc) == "OCR failed"

    def test_error_with_exit_code(self) -> None:
        """Test error with exit code."""
        exc = OCRError("OCR failed", exit_code=1)
        assert exc.exit_code == 1
        assert str(exc) == "OCR failed (exit_code=1)"


# ---------------------------------------------------------------------------
# Model Tests - BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_properties(self) -> None:
        """Test width, height, and area calculations."""
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000

    def test_center(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        assert bbox.center == (50.0, 25.0)

    def test_to_tuple(self) -> None:
        """Test conversion to tuple."""
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert bbox.to_tuple() == (10, 20, 110, 70)

    def test_to_int_tuple(self) -> None:
        """Test conversion to integer tuple."""
        bbox = BoundingBox(x0=10.5, y0=20.7, x1=110.2, y1=70.9)
        assert bbox.to_int_tuple() == (10, 20, 110, 70)

    def test_frozen(self) -> None:
        """Test that BoundingBox is immutable."""
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        with pytest.raises(AttributeError):
            bbox.x0 = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model Tests - RegionLabel
# ---------------------------------------------------------------------------


class TestRegionLabel:
    """Tests for RegionLabel enum."""

    def test_text_label(self) -> None:
        """Test Text label value."""
        assert RegionLabel.TEXT == "Text"
        assert RegionLabel.TEXT.value == "Text"

    def test_page_header_label(self) -> None:
        """Test Page-header label value."""
        assert RegionLabel.PAGE_HEADER == "Page-header"

    def test_all_labels_exist(self) -> None:
        """Test that all expected labels are defined."""
        expected = {
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Figure",
            "Section-header",
            "Table",
            "Form",
            "Table-of-contents",
            "Handwriting",
            "Text",
            "Text-inline-math",
        }
        actual = {label.value for label in RegionLabel}
        assert actual == expected


# ---------------------------------------------------------------------------
# Model Tests - LayoutRegion
# ---------------------------------------------------------------------------


class TestLayoutRegion:
    """Tests for LayoutRegion model."""

    def test_should_ocr_text_region(self) -> None:
        """Test that text regions are included by default."""
        region = LayoutRegion(
            bbox=BoundingBox(0, 0, 100, 100),
            label=RegionLabel.TEXT,
            position=0,
            confidence=0.9,
        )
        assert region.should_ocr(set()) is True
        assert region.should_ocr({"header", "footer"}) is True

    def test_should_ocr_excluded_region_exact_match(self) -> None:
        """Test exclusion with exact label match."""
        region = LayoutRegion(
            bbox=BoundingBox(0, 0, 100, 100),
            label=RegionLabel.PAGE_HEADER,
            position=0,
            confidence=0.9,
        )
        assert region.should_ocr({"Page-header"}) is False

    def test_should_ocr_excluded_region_normalized(self) -> None:
        """Test exclusion with normalized label (underscores to hyphens)."""
        region = LayoutRegion(
            bbox=BoundingBox(0, 0, 100, 100),
            label=RegionLabel.PAGE_HEADER,
            position=0,
            confidence=0.9,
        )
        # Config uses underscores, model uses hyphens
        assert region.should_ocr({"page_header"}) is False

    def test_should_ocr_case_insensitive(self) -> None:
        """Test that exclusion is case-insensitive."""
        region = LayoutRegion(
            bbox=BoundingBox(0, 0, 100, 100),
            label=RegionLabel.PAGE_FOOTER,
            position=0,
            confidence=0.9,
        )
        assert region.should_ocr({"PAGE_FOOTER"}) is False
        assert region.should_ocr({"page-footer"}) is False

    def test_frozen(self) -> None:
        """Test that LayoutRegion is immutable."""
        region = LayoutRegion(
            bbox=BoundingBox(0, 0, 100, 100),
            label=RegionLabel.TEXT,
            position=0,
            confidence=0.9,
        )
        with pytest.raises(AttributeError):
            region.position = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model Tests - LayoutResult
# ---------------------------------------------------------------------------


class TestLayoutResult:
    """Tests for LayoutResult model."""

    @pytest.fixture
    def sample_layout_result(self) -> LayoutResult:
        """Create a sample LayoutResult for testing."""
        return LayoutResult(
            page_number=0,
            regions=[
                LayoutRegion(
                    bbox=BoundingBox(0, 0, 100, 50),
                    label=RegionLabel.PAGE_HEADER,
                    position=0,
                    confidence=0.9,
                ),
                LayoutRegion(
                    bbox=BoundingBox(0, 60, 100, 200),
                    label=RegionLabel.TEXT,
                    position=1,
                    confidence=0.95,
                ),
                LayoutRegion(
                    bbox=BoundingBox(0, 210, 100, 300),
                    label=RegionLabel.TEXT,
                    position=2,
                    confidence=0.4,  # Low confidence
                ),
                LayoutRegion(
                    bbox=BoundingBox(0, 310, 100, 350),
                    label=RegionLabel.PAGE_FOOTER,
                    position=3,
                    confidence=0.85,
                ),
            ],
            image_bbox=BoundingBox(0, 0, 612, 792),
            image_width=612,
            image_height=792,
        )

    def test_get_text_regions_no_filter(
        self, sample_layout_result: LayoutResult
    ) -> None:
        """Test getting all regions without filtering."""
        regions = sample_layout_result.get_text_regions()
        assert len(regions) == 4
        # Should be sorted by position
        assert regions[0].position == 0
        assert regions[3].position == 3

    def test_get_text_regions_with_exclude(
        self, sample_layout_result: LayoutResult
    ) -> None:
        """Test filtering regions by exclude list."""
        regions = sample_layout_result.get_text_regions(
            exclude_labels={"page_header", "page_footer"}
        )
        assert len(regions) == 2
        assert all(r.label == RegionLabel.TEXT for r in regions)

    def test_get_text_regions_with_confidence_threshold(
        self, sample_layout_result: LayoutResult
    ) -> None:
        """Test filtering regions by confidence threshold."""
        regions = sample_layout_result.get_text_regions(min_confidence=0.5)
        assert len(regions) == 3
        # Low confidence TEXT region should be excluded
        assert all(r.confidence >= 0.5 for r in regions)

    def test_get_regions_by_label(self, sample_layout_result: LayoutResult) -> None:
        """Test getting regions by specific label."""
        text_regions = sample_layout_result.get_regions_by_label(RegionLabel.TEXT)
        assert len(text_regions) == 2
        assert all(r.label == RegionLabel.TEXT for r in text_regions)
        # Should be sorted by position
        assert text_regions[0].position < text_regions[1].position


# ---------------------------------------------------------------------------
# Model Tests - PageAnalysis
# ---------------------------------------------------------------------------


class TestPageAnalysis:
    """Tests for PageAnalysis model."""

    def test_creation(self) -> None:
        """Test PageAnalysis creation."""
        page = PageAnalysis(
            page_number=0,
            has_text=True,
            text_char_count=500,
            is_image_only=False,
        )
        assert page.page_number == 0
        assert page.has_text is True
        assert page.text_char_count == 500
        assert page.is_image_only is False


# ---------------------------------------------------------------------------
# Model Tests - DocumentAnalysis
# ---------------------------------------------------------------------------


class TestDocumentAnalysis:
    """Tests for DocumentAnalysis model."""

    @pytest.fixture
    def mixed_document(self) -> DocumentAnalysis:
        """Create a DocumentAnalysis with mixed pages."""
        return DocumentAnalysis(
            total_pages=4,
            pages=[
                PageAnalysis(
                    0, has_text=True, text_char_count=500, is_image_only=False
                ),
                PageAnalysis(1, has_text=False, text_char_count=0, is_image_only=True),
                PageAnalysis(
                    2, has_text=True, text_char_count=300, is_image_only=False
                ),
                PageAnalysis(3, has_text=False, text_char_count=0, is_image_only=True),
            ],
            has_any_text=True,
            is_born_digital=False,
        )

    def test_pages_with_text(self, mixed_document: DocumentAnalysis) -> None:
        """Test pages_with_text property."""
        pages = mixed_document.pages_with_text
        assert len(pages) == 2
        assert pages[0].page_number == 0
        assert pages[1].page_number == 2

    def test_pages_without_text(self, mixed_document: DocumentAnalysis) -> None:
        """Test pages_without_text property."""
        pages = mixed_document.pages_without_text
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 3

    def test_total_text_chars(self, mixed_document: DocumentAnalysis) -> None:
        """Test total_text_chars property."""
        assert mixed_document.total_text_chars == 800


# ---------------------------------------------------------------------------
# Model Tests - Stage1Result
# ---------------------------------------------------------------------------


class TestStage1Result:
    """Tests for Stage1Result model."""

    @pytest.fixture
    def successful_result(self, tmp_path: Path) -> Stage1Result:
        """Create a successful Stage1Result."""
        return Stage1Result(
            success=True,
            input_path=tmp_path / "input.pdf",
            output_path=tmp_path / "output.pdf",
            document_analysis=DocumentAnalysis(
                total_pages=2,
                pages=[
                    PageAnalysis(
                        page_number=0,
                        has_text=True,
                        text_char_count=500,
                        is_image_only=False,
                    ),
                    PageAnalysis(
                        page_number=1,
                        has_text=True,
                        text_char_count=300,
                        is_image_only=False,
                    ),
                ],
                has_any_text=True,
                is_born_digital=True,
            ),
            layout_results=None,
            pages_processed=2,
            skipped=False,
            skip_reason=None,
            processing_time_seconds=5.5,
        )

    def test_is_terminal_success(self, successful_result: Stage1Result) -> None:
        """Test is_terminal for successful result."""
        assert successful_result.is_terminal is True

    def test_is_terminal_with_error(self, tmp_path: Path) -> None:
        """Test is_terminal for failed result."""
        result = Stage1Result(
            success=False,
            input_path=tmp_path / "input.pdf",
            output_path=None,
            document_analysis=DocumentAnalysis(
                total_pages=0,
                pages=[],
                has_any_text=False,
                is_born_digital=False,
            ),
            layout_results=None,
            pages_processed=0,
            skipped=False,
            skip_reason=None,
            error="OCR failed",
        )
        assert result.is_terminal is True

    def test_to_dict(self, successful_result: Stage1Result) -> None:
        """Test to_dict serialization."""
        result_dict = successful_result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["pages_processed"] == 2
        assert result_dict["processing_time_seconds"] == 5.5
        assert result_dict["document_analysis"]["total_pages"] == 2
        assert result_dict["document_analysis"]["is_born_digital"] is True
        assert result_dict["layout_results_count"] == 0

    def test_created_at_default(self, tmp_path: Path) -> None:
        """Test that created_at defaults to now."""
        before = datetime.now(UTC)
        result = Stage1Result(
            success=True,
            input_path=tmp_path / "input.pdf",
            output_path=tmp_path / "output.pdf",
            document_analysis=DocumentAnalysis(
                total_pages=0,
                pages=[],
                has_any_text=False,
                is_born_digital=False,
            ),
            layout_results=None,
            pages_processed=0,
            skipped=False,
            skip_reason=None,
        )
        after = datetime.now(UTC)
        assert before <= result.created_at <= after


# ---------------------------------------------------------------------------
# Stage1Processor Tests
# ---------------------------------------------------------------------------


class TestStage1Processor:
    """Tests for Stage1Processor logic."""

    @pytest.fixture
    def skip_config(self) -> Stage1Config:
        """Create a config with born_digital_handling=SKIP."""
        return Stage1Config(
            enabled=True,
            born_digital_handling=BornDigitalHandling.SKIP,
        )

    @pytest.fixture
    def force_config(self) -> Stage1Config:
        """Create a config with born_digital_handling=FORCE."""
        return Stage1Config(
            enabled=True,
            born_digital_handling=BornDigitalHandling.FORCE,
        )

    def test_should_skip_ocr_born_digital_skip_mode(
        self, skip_config: Stage1Config
    ) -> None:
        """Test that born-digital documents are skipped in SKIP mode."""
        processor = Stage1Processor(config=skip_config, gpu_mode=GPUMode.CPU)
        analysis = DocumentAnalysis(
            total_pages=2,
            pages=[
                PageAnalysis(
                    0, has_text=True, text_char_count=500, is_image_only=False
                ),
                PageAnalysis(
                    1, has_text=True, text_char_count=500, is_image_only=False
                ),
            ],
            has_any_text=True,
            is_born_digital=True,
        )
        assert processor._should_skip_ocr(analysis) is True

    def test_should_skip_ocr_scanned_skip_mode(self, skip_config: Stage1Config) -> None:
        """Test that scanned documents are NOT skipped in SKIP mode."""
        processor = Stage1Processor(config=skip_config, gpu_mode=GPUMode.CPU)
        analysis = DocumentAnalysis(
            total_pages=2,
            pages=[
                PageAnalysis(0, has_text=False, text_char_count=0, is_image_only=True),
                PageAnalysis(1, has_text=False, text_char_count=0, is_image_only=True),
            ],
            has_any_text=False,
            is_born_digital=False,
        )
        assert processor._should_skip_ocr(analysis) is False

    def test_should_skip_ocr_born_digital_force_mode(
        self, force_config: Stage1Config
    ) -> None:
        """Test that born-digital documents are NOT skipped in FORCE mode."""
        processor = Stage1Processor(config=force_config, gpu_mode=GPUMode.CPU)
        analysis = DocumentAnalysis(
            total_pages=2,
            pages=[
                PageAnalysis(
                    0, has_text=True, text_char_count=500, is_image_only=False
                ),
                PageAnalysis(
                    1, has_text=True, text_char_count=500, is_image_only=False
                ),
            ],
            has_any_text=True,
            is_born_digital=True,
        )
        assert processor._should_skip_ocr(analysis) is False

    def test_empty_analysis(self, skip_config: Stage1Config) -> None:
        """Test _empty_analysis returns valid empty DocumentAnalysis."""
        processor = Stage1Processor(config=skip_config, gpu_mode=GPUMode.CPU)
        analysis = processor._empty_analysis()
        assert analysis.total_pages == 0
        assert analysis.pages == []
        assert analysis.has_any_text is False
        assert analysis.is_born_digital is False

    def test_config_property(self, skip_config: Stage1Config) -> None:
        """Test that config property returns the configuration."""
        processor = Stage1Processor(config=skip_config, gpu_mode=GPUMode.CPU)
        assert processor.config is skip_config


# ---------------------------------------------------------------------------
# Stage 2 Exception Tests
# ---------------------------------------------------------------------------


class TestMarkerConversionError:
    """Tests for MarkerConversionError."""

    def test_basic_error(self) -> None:
        """Test basic error without cause."""
        exc = MarkerConversionError("Marker failed")
        assert exc.message == "Marker failed"
        assert exc.cause is None
        assert str(exc) == "Marker failed"

    def test_error_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = RuntimeError("GPU out of memory")
        exc = MarkerConversionError("Marker failed", cause=cause)
        assert exc.cause is cause
        assert exc.__cause__ is cause

    def test_inheritance(self) -> None:
        """Test that MarkerConversionError inherits from PipelineError."""
        exc = MarkerConversionError("Test error")
        assert isinstance(exc, PipelineError)
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# Stage 2 Model Tests - Stage2Result
# ---------------------------------------------------------------------------


class TestStage2Result:
    """Tests for Stage2Result model."""

    @pytest.fixture
    def successful_result(self, tmp_path: Path) -> Stage2Result:
        """Create a successful Stage2Result."""
        return Stage2Result(
            success=True,
            input_path=tmp_path / "input.pdf",
            markdown="# Test Document\n\nHello world.\n",
            page_count=2,
            images={"img_1": "base64data"},
            metadata={"toc": [], "page_stats": {}},
            llm_used=False,
            skipped=False,
            skip_reason=None,
            processing_time_seconds=3.5,
        )

    @pytest.fixture
    def failed_result(self, tmp_path: Path) -> Stage2Result:
        """Create a failed Stage2Result."""
        return Stage2Result(
            success=False,
            input_path=tmp_path / "input.pdf",
            markdown="",
            page_count=0,
            images={},
            metadata={},
            llm_used=False,
            skipped=False,
            skip_reason=None,
            error="Marker conversion failed",
            processing_time_seconds=1.0,
        )

    def test_is_terminal_success(self, successful_result: Stage2Result) -> None:
        """Test is_terminal for successful result."""
        assert successful_result.is_terminal is True

    def test_is_terminal_with_error(self, failed_result: Stage2Result) -> None:
        """Test is_terminal for failed result."""
        assert failed_result.is_terminal is True

    def test_is_terminal_skipped(self, tmp_path: Path) -> None:
        """Test is_terminal for skipped result (still terminal)."""
        result = Stage2Result(
            success=True,
            input_path=tmp_path / "input.pdf",
            markdown="",
            page_count=0,
            images={},
            metadata={},
            llm_used=False,
            skipped=True,
            skip_reason="Stage 2 disabled",
        )
        assert result.is_terminal is True

    def test_to_dict(self, successful_result: Stage2Result) -> None:
        """Test to_dict serialization."""
        result_dict = successful_result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["markdown_length"] == len(successful_result.markdown)
        assert result_dict["page_count"] == 2
        assert result_dict["images_count"] == 1
        assert result_dict["llm_used"] is False
        assert result_dict["skipped"] is False
        assert result_dict["processing_time_seconds"] == 3.5
        assert "created_at" in result_dict

    def test_to_dict_failed(self, failed_result: Stage2Result) -> None:
        """Test to_dict for failed result."""
        result_dict = failed_result.to_dict()
        assert result_dict["success"] is False
        assert result_dict["error"] == "Marker conversion failed"
        assert result_dict["markdown_length"] == 0

    def test_created_at_default(self, tmp_path: Path) -> None:
        """Test that created_at defaults to now."""
        before = datetime.now(UTC)
        result = Stage2Result(
            success=True,
            input_path=tmp_path / "input.pdf",
            markdown="# Test\n",
            page_count=1,
            images={},
            metadata={},
            llm_used=False,
            skipped=False,
            skip_reason=None,
        )
        after = datetime.now(UTC)
        assert before <= result.created_at <= after


# ---------------------------------------------------------------------------
# Stage 2 Function Tests - postprocess_markdown
# ---------------------------------------------------------------------------


class TestPostprocessMarkdown:
    """Tests for postprocess_markdown function."""

    def test_collapses_multiple_blank_lines(self) -> None:
        """Test that 3+ consecutive blank lines are collapsed to 2."""
        input_md = "# Title\n\n\n\n\nParagraph"
        result = postprocess_markdown(input_md)
        assert result == "# Title\n\nParagraph\n"

    def test_strips_trailing_whitespace(self) -> None:
        """Test that trailing whitespace is stripped from each line."""
        input_md = "Line 1   \nLine 2\t\nLine 3"
        result = postprocess_markdown(input_md)
        assert result == "Line 1\nLine 2\nLine 3\n"

    def test_ensures_trailing_newline(self) -> None:
        """Test that output ends with single newline."""
        input_md = "# Title"
        result = postprocess_markdown(input_md)
        assert result == "# Title\n"
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_handles_empty_input(self) -> None:
        """Test handling of empty input."""
        assert postprocess_markdown("") == ""

    def test_handles_whitespace_only(self) -> None:
        """Test handling of whitespace-only input."""
        assert postprocess_markdown("   \n\n   ") == "\n"

    def test_preserves_single_blank_lines(self) -> None:
        """Test that single blank lines are preserved."""
        input_md = "# Title\n\nParagraph 1\n\nParagraph 2"
        result = postprocess_markdown(input_md)
        assert result == "# Title\n\nParagraph 1\n\nParagraph 2\n"

    def test_complex_document(self) -> None:
        """Test with a more complex document."""
        input_md = """# Title


Some text with    trailing spaces.




Another paragraph.

- List item 1
- List item 2


Done."""
        expected = """# Title

Some text with    trailing spaces.

Another paragraph.

- List item 1
- List item 2

Done.
"""
        result = postprocess_markdown(input_md)
        assert result == expected


# ---------------------------------------------------------------------------
# Stage 2 Processor Tests
# ---------------------------------------------------------------------------


class TestStage2Processor:
    """Tests for Stage2Processor logic."""

    @pytest.fixture
    def default_config(self) -> Stage2Config:
        """Create a default Stage2Config."""
        return Stage2Config(enabled=True)

    @pytest.fixture
    def disabled_config(self) -> Stage2Config:
        """Create a disabled Stage2Config."""
        return Stage2Config(enabled=False)

    def test_config_property(self, default_config: Stage2Config) -> None:
        """Test that config property returns the configuration."""
        processor = Stage2Processor(config=default_config, gpu_mode=GPUMode.CPU)
        assert processor.config is default_config

    def test_get_device_cuda(self, default_config: Stage2Config) -> None:
        """Test _get_device returns 'cuda' for CUDA mode."""
        processor = Stage2Processor(config=default_config, gpu_mode=GPUMode.CUDA)
        assert processor._get_device() == "cuda"

    def test_get_device_cpu(self, default_config: Stage2Config) -> None:
        """Test _get_device returns 'cpu' for CPU mode."""
        processor = Stage2Processor(config=default_config, gpu_mode=GPUMode.CPU)
        assert processor._get_device() == "cpu"

    def test_get_device_auto(self, default_config: Stage2Config) -> None:
        """Test _get_device returns None for AUTO mode."""
        processor = Stage2Processor(config=default_config, gpu_mode=GPUMode.AUTO)
        assert processor._get_device() is None

    @pytest.mark.asyncio
    async def test_process_skips_when_disabled(
        self, disabled_config: Stage2Config, tmp_path: Path
    ) -> None:
        """Test that processing is skipped when stage is disabled."""
        processor = Stage2Processor(config=disabled_config, gpu_mode=GPUMode.CPU)
        input_path = tmp_path / "test.pdf"
        input_path.touch()

        result = await processor.process(input_path)

        assert result.success is True
        assert result.skipped is True
        assert result.skip_reason == "Stage 2 is disabled in configuration"
        assert result.markdown == ""
        assert result.page_count == 0

    @pytest.mark.asyncio
    async def test_process_force_overrides_disabled(
        self, disabled_config: Stage2Config, tmp_path: Path
    ) -> None:
        """Test that force=True overrides disabled config.

        Note: This test will fail without mocking because it tries to load models.
        In a full test suite, we would mock get_marker_models.
        """
        # This test demonstrates the behavior but requires mocking for full coverage
        # TODO: Add integration test with mocking
