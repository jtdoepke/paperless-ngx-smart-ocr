"""Stage 2: Markdown Pipeline.

This module implements the Markdown conversion pipeline that:
1. Converts PDFs to structured Markdown using Marker
2. Post-processes and validates the output
3. Optionally uses LLM for complex elements (configurable, off by default)

The pipeline respects configuration for LLM usage and produces clean,
structured Markdown preserving document structure.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any

from paperless_ngx_smart_ocr.config import GPUMode, LLMProvider, Stage2Config
from paperless_ngx_smart_ocr.observability import get_logger
from paperless_ngx_smart_ocr.pipeline.exceptions import (
    MarkerConversionError,
    PipelineError,
)
from paperless_ngx_smart_ocr.pipeline.models import Stage2Result


if TYPE_CHECKING:
    from pathlib import Path

    from paperless_ngx_smart_ocr.config import MarkerConfig


__all__ = [
    "Stage2Processor",
    "clear_marker_models",
    "get_marker_models",
    "postprocess_markdown",
    "process_stage2",
]


# ---------------------------------------------------------------------------
# Module-level model cache (Marker models are expensive to load)
# ---------------------------------------------------------------------------

_marker_models: dict[str, Any] | None = None
_marker_lock = asyncio.Lock()


async def get_marker_models(
    device: str | None = None,
    dtype: Any | None = None,  # noqa: ANN401
) -> dict[str, Any]:
    """Get cached Marker models, loading them lazily if needed.

    Marker models (Surya layout, recognition, etc.) are large and expensive
    to load. This function caches them in a module-level singleton, loading
    them on first use and reusing for subsequent conversions.

    Args:
        device: Optional device override ('cuda', 'cpu', etc.).
        dtype: Optional dtype override for model precision.

    Returns:
        Dictionary of loaded Marker models ready for PdfConverter.

    Raises:
        MarkerConversionError: If models fail to load.

    Note:
        Thread-safe via asyncio.Lock. Models are loaded in a thread pool
        to avoid blocking the event loop.
    """
    global _marker_models  # noqa: PLW0603

    async with _marker_lock:
        if _marker_models is not None:
            return _marker_models

        logger = get_logger(__name__)
        logger.info("loading_marker_models", device=device)

        try:
            from marker.models import create_model_dict

            # Load models in thread pool (CPU-bound operation)
            loaded_models: dict[str, Any] = await asyncio.to_thread(
                create_model_dict,
                device=device,
                dtype=dtype,
            )
            _marker_models = loaded_models

            logger.info("marker_models_loaded")
            return loaded_models  # noqa: TRY300

        except Exception as exc:
            msg = f"Failed to load Marker models: {exc}"
            raise MarkerConversionError(msg, cause=exc) from exc


def clear_marker_models() -> None:
    """Clear cached Marker models to free GPU/CPU memory.

    Call this when you no longer need the models and want to release
    resources. Models will be reloaded on next conversion.
    """
    global _marker_models  # noqa: PLW0603
    _marker_models = None

    logger = get_logger(__name__)
    logger.info("marker_models_cleared")


# ---------------------------------------------------------------------------
# LLM Service Factory
# ---------------------------------------------------------------------------


def create_llm_service(config: MarkerConfig) -> Any | None:  # noqa: ANN401
    """Create Marker LLM service from configuration.

    Maps the LLMProvider enum to Marker's service classes and configures
    them with the appropriate API keys and model names.

    Args:
        config: Marker configuration containing LLM settings.

    Returns:
        Configured LLM service instance, or None if LLM is disabled.

    Raises:
        MarkerConversionError: If LLM service creation fails.
    """
    if not config.use_llm:
        return None

    llm_config = config.llm
    logger = get_logger(__name__)

    try:
        match llm_config.provider:
            case LLMProvider.OPENAI:
                from marker.services.openai import OpenAIService

                service_config = {
                    "openai_api_key": llm_config.api_key,
                    "openai_model": llm_config.model,
                }
                logger.debug(
                    "creating_llm_service",
                    provider="openai",
                    model=llm_config.model,
                )
                return OpenAIService(service_config)

            case LLMProvider.ANTHROPIC:
                from marker.services.claude import ClaudeService

                service_config = {
                    "claude_api_key": llm_config.api_key,
                    "claude_model_name": llm_config.model,
                }
                logger.debug(
                    "creating_llm_service",
                    provider="anthropic",
                    model=llm_config.model,
                )
                return ClaudeService(service_config)

            case LLMProvider.OLLAMA:
                from marker.services.ollama import OllamaService

                service_config = {
                    "ollama_model": llm_config.model,
                    # ollama_base_url defaults to localhost:11434
                }
                logger.debug(
                    "creating_llm_service",
                    provider="ollama",
                    model=llm_config.model,
                )
                return OllamaService(service_config)

    except MarkerConversionError:
        raise

    except Exception as exc:
        msg = f"Failed to create LLM service: {exc}"
        raise MarkerConversionError(msg, cause=exc) from exc


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------


def postprocess_markdown(markdown: str) -> str:
    """Clean up and normalize Marker's Markdown output.

    Performs the following cleanup operations:
    1. Collapses 3+ consecutive blank lines to 2
    2. Strips trailing whitespace from each line
    3. Ensures a single trailing newline

    Args:
        markdown: Raw Markdown output from Marker.

    Returns:
        Cleaned Markdown string.
    """
    if not markdown:
        return ""

    # Collapse 3+ consecutive blank lines to 2
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in markdown.split("\n")]
    markdown = "\n".join(lines)

    # Ensure single trailing newline
    return markdown.strip() + "\n"


# ---------------------------------------------------------------------------
# Stage 2 Processor
# ---------------------------------------------------------------------------


class Stage2Processor:
    """Markdown converter for Stage 2 of the pipeline.

    Handles the complete Markdown conversion workflow:
    1. Load/cache Marker models
    2. Convert PDF to Markdown using Marker
    3. Post-process output
    4. Return structured result

    The processor can optionally use LLM assistance for complex elements
    like tables, equations, and images (disabled by default).

    Attributes:
        config: Stage 2 configuration settings.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings

        settings = get_settings()
        processor = Stage2Processor(
            config=settings.pipeline.stage2,
            gpu_mode=settings.gpu.enabled,
        )

        result = await processor.process(pdf_path)
        if result.success:
            print(f"Converted {result.page_count} pages")
            print(result.markdown[:500])
        else:
            print(f"Failed: {result.error}")
        ```
    """

    def __init__(
        self,
        *,
        config: Stage2Config,
        gpu_mode: GPUMode = GPUMode.AUTO,
    ) -> None:
        """Initialize the Stage 2 processor.

        Args:
            config: Stage 2 configuration from settings.
            gpu_mode: GPU mode for model loading.
        """
        self._config = config
        self._gpu_mode = gpu_mode
        self._logger = get_logger(__name__)

    @property
    def config(self) -> Stage2Config:
        """Get the Stage 2 configuration."""
        return self._config

    async def process(
        self,
        input_path: Path,
        *,
        force: bool = False,
    ) -> Stage2Result:
        """Process a document through Stage 2 Markdown conversion.

        Converts a PDF to Markdown using Marker, optionally with LLM
        assistance. The output is post-processed for consistency.

        Args:
            input_path: Path to input PDF file.
            force: If True, process even if stage is disabled in config.

        Returns:
            Stage2Result with conversion outcome, Markdown content, and metadata.

        Raises:
            PipelineError: If processing fails with an unrecoverable error.

        Note:
            This method is async to integrate with the job queue. The
            underlying Marker conversion runs in a thread pool.
        """
        start_time = time.monotonic()
        self._logger.info(
            "stage2_starting",
            input=str(input_path),
            use_llm=self._config.marker.use_llm,
            force=force,
        )

        # Check if stage is enabled
        if not self._config.enabled and not force:
            processing_time = time.monotonic() - start_time
            self._logger.info(
                "stage2_skipped",
                input=str(input_path),
                reason="disabled",
            )
            return Stage2Result(
                success=True,
                input_path=input_path,
                markdown="",
                page_count=0,
                images={},
                metadata={},
                llm_used=False,
                skipped=True,
                skip_reason="Stage 2 is disabled in configuration",
                processing_time_seconds=processing_time,
            )

        try:
            # Step 1: Load/get cached models
            device = self._get_device()
            models = await get_marker_models(device=device)

            # Step 2: Create LLM service if configured
            llm_service = create_llm_service(self._config.marker)

            # Step 3: Run Marker conversion
            markdown, images, metadata, page_count = await self._run_marker(
                input_path,
                models,
                llm_service,
            )

            # Step 4: Post-process Markdown
            markdown = postprocess_markdown(markdown)

            processing_time = time.monotonic() - start_time
            self._logger.info(
                "stage2_completed",
                input=str(input_path),
                page_count=page_count,
                markdown_length=len(markdown),
                images_count=len(images),
                llm_used=self._config.marker.use_llm,
                processing_time_seconds=processing_time,
            )

            return Stage2Result(
                success=True,
                input_path=input_path,
                markdown=markdown,
                page_count=page_count,
                images=images,
                metadata=metadata,
                llm_used=self._config.marker.use_llm,
                skipped=False,
                skip_reason=None,
                processing_time_seconds=processing_time,
            )

        except MarkerConversionError as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage2_conversion_failed",
                input=str(input_path),
                error=str(exc),
            )
            return Stage2Result(
                success=False,
                input_path=input_path,
                markdown="",
                page_count=0,
                images={},
                metadata={},
                llm_used=self._config.marker.use_llm,
                skipped=False,
                skip_reason=None,
                error=str(exc),
                processing_time_seconds=processing_time,
            )

        except PipelineError as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage2_failed",
                input=str(input_path),
                error=str(exc),
            )
            return Stage2Result(
                success=False,
                input_path=input_path,
                markdown="",
                page_count=0,
                images={},
                metadata={},
                llm_used=self._config.marker.use_llm,
                skipped=False,
                skip_reason=None,
                error=str(exc),
                processing_time_seconds=processing_time,
            )

        except Exception as exc:
            processing_time = time.monotonic() - start_time
            self._logger.exception(
                "stage2_unexpected_error",
                input=str(input_path),
                error=str(exc),
            )
            # Re-raise unexpected exceptions as PipelineError
            msg = f"Stage 2 processing failed unexpectedly: {exc}"
            raise PipelineError(msg) from exc

    def _get_device(self) -> str | None:
        """Get the device string based on GPU mode.

        Returns:
            Device string ('cuda', 'cpu', or None for auto-detection).
        """
        match self._gpu_mode:
            case GPUMode.CUDA:
                return "cuda"
            case GPUMode.CPU:
                return "cpu"
            case GPUMode.AUTO:
                return None

    async def _run_marker(
        self,
        input_path: Path,
        models: dict[str, Any],
        llm_service: Any | None,  # noqa: ANN401
    ) -> tuple[str, dict[str, str], dict[str, Any], int]:
        """Run Marker conversion in a thread pool.

        Args:
            input_path: Path to PDF file.
            models: Pre-loaded Marker models.
            llm_service: Optional LLM service for complex elements.

        Returns:
            Tuple of (markdown, images, metadata, page_count).

        Raises:
            MarkerConversionError: If conversion fails.
        """
        self._logger.debug("running_marker", path=str(input_path))

        try:
            from marker.converters.pdf import PdfConverter

            # Create converter with models and optional LLM
            config = {"use_llm": self._config.marker.use_llm}

            def run_conversion() -> tuple[str, dict[str, str], dict[str, Any], int]:
                converter = PdfConverter(
                    artifact_dict=models,
                    processor_list=None,
                    renderer=None,  # Uses default MarkdownRenderer
                    llm_service=llm_service,
                    config=config,
                )

                # Run conversion - returns MarkdownOutput
                output = converter(str(input_path))

                return (
                    output.markdown,
                    output.images,
                    output.metadata,
                    converter.page_count or 0,
                )

            # Run in thread pool (CPU/GPU bound operation)
            result = await asyncio.to_thread(run_conversion)

            self._logger.debug(
                "marker_completed",
                path=str(input_path),
                page_count=result[3],
            )

            return result  # noqa: TRY300

        except Exception as exc:
            msg = f"Marker conversion failed: {exc}"
            raise MarkerConversionError(msg, cause=exc) from exc


# ---------------------------------------------------------------------------
# Convenience Function
# ---------------------------------------------------------------------------


async def process_stage2(
    input_path: Path,
    *,
    config: Stage2Config,
    gpu_mode: GPUMode = GPUMode.AUTO,
    force: bool = False,
) -> Stage2Result:
    """Convenience function to process a document through Stage 2.

    Creates a Stage2Processor and processes the document in a single call.

    Args:
        input_path: Path to input PDF file.
        config: Stage 2 configuration from settings.
        gpu_mode: GPU mode for model loading.
        force: If True, process even if stage is disabled.

    Returns:
        Stage2Result with conversion outcome.

    Example:
        ```python
        from paperless_ngx_smart_ocr.config import get_settings

        settings = get_settings()
        result = await process_stage2(
            input_path,
            config=settings.pipeline.stage2,
            gpu_mode=settings.gpu.enabled,
        )
        ```
    """
    processor = Stage2Processor(config=config, gpu_mode=gpu_mode)
    return await processor.process(input_path, force=force)
