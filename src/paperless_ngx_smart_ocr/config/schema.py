"""Configuration schema models for paperless-ngx-smart-ocr.

This module defines Pydantic models for all configuration sections.
These models are used by the Settings class to validate and type-check
configuration loaded from YAML files and environment variables.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 - needed at runtime for Pydantic
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


__all__ = [
    # Configuration models
    "AutoProcessingConfig",
    # Enums
    "BornDigitalHandling",
    # Base model
    "ConfigBaseModel",
    "ContentMode",
    "GPUConfig",
    "GPUMode",
    "IntegrationConfig",
    "JobsConfig",
    "LLMConfig",
    "LLMProvider",
    "LayoutDetectionConfig",
    "LogFormat",
    "LogLevel",
    "LoggingConfig",
    "MarkerConfig",
    "MetricsConfig",
    "OCRmyPDFConfig",
    "ObservabilityConfig",
    "PaperlessConfig",
    "PipelineConfig",
    "PollingConfig",
    "PostConsumeConfig",
    "Stage1Config",
    "Stage2Config",
    "TagsConfig",
    "ThemeMode",
    "TracingConfig",
    "WebConfig",
    "WebhookConfig",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BornDigitalHandling(StrEnum):
    """How to handle documents that already have a text layer.

    Attributes:
        SKIP: Skip OCR if text is detected in the document.
        FORCE: Always perform OCR, replacing any existing text layer.
    """

    SKIP = "skip"
    FORCE = "force"


class ContentMode(StrEnum):
    """How to handle existing document content in Stage 2.

    Attributes:
        REPLACE: Replace the content field entirely with Markdown.
        APPEND: Append Markdown below the existing content.
    """

    REPLACE = "replace"
    APPEND = "append"


class GPUMode(StrEnum):
    """GPU acceleration mode for Surya and Marker.

    Attributes:
        AUTO: Automatically detect and use GPU if available.
        CUDA: Force NVIDIA CUDA GPU usage.
        CPU: Disable GPU, use CPU only.
    """

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class LogFormat(StrEnum):
    """Log output format.

    Attributes:
        JSON: Structured JSON logging (for machine parsing).
        CONSOLE: Human-readable console output with colors.
    """

    JSON = "json"
    CONSOLE = "console"


class LogLevel(StrEnum):
    """Log verbosity level.

    Attributes:
        DEBUG: Detailed debugging information.
        INFO: General operational information.
        WARNING: Warning messages for potential issues.
        ERROR: Error messages for failures.
        CRITICAL: Critical errors that may cause shutdown.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ThemeMode(StrEnum):
    """Web UI theme mode.

    Attributes:
        AUTO: Follow system preference.
        DARK: Always use dark theme.
        LIGHT: Always use light theme.
    """

    AUTO = "auto"
    DARK = "dark"
    LIGHT = "light"


class LLMProvider(StrEnum):
    """LLM provider for Marker's optional LLM assistance.

    Attributes:
        OPENAI: OpenAI API (GPT models).
        ANTHROPIC: Anthropic API (Claude models).
        OLLAMA: Local Ollama server.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Base Configuration Model
# ---------------------------------------------------------------------------


class ConfigBaseModel(BaseModel):
    """Base model for all configuration sections.

    Uses stricter settings than API models to catch configuration typos:
    - extra="forbid" raises errors for unknown fields
    - validate_default=True ensures defaults are validated
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )


# ---------------------------------------------------------------------------
# Paperless-ngx Connection
# ---------------------------------------------------------------------------


class PaperlessConfig(ConfigBaseModel):
    """Paperless-ngx connection configuration.

    Either `token` or `token_file` must be provided. If both are set,
    `token` takes precedence. Environment variable interpolation is
    supported in the `token` field using ${VAR} syntax.

    Attributes:
        url: Base URL of the Paperless-ngx instance.
        token: API authentication token (supports ${VAR} interpolation).
        token_file: Path to a file containing the API token.
        archive_dir: Mount point for the paperless-ngx archive
            directory. Required (along with ``database_url``) to
            enable the "Replace PDF" feature.
        database_url: PostgreSQL connection URL for updating
            ``archive_checksum`` after archive replacement. Required
            (along with ``archive_dir``) to enable "Replace PDF".
    """

    url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the Paperless-ngx instance",
    )
    token: str | None = Field(
        default=None,
        description="API authentication token (supports ${VAR} interpolation)",
    )
    token_file: Path | None = Field(
        default=None,
        description="Path to file containing the API token",
    )
    archive_dir: Path | None = Field(
        default=None,
        description="Mount point for paperless-ngx archive directory",
    )
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for archive_checksum updates",
    )

    @field_validator("url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash from URL to avoid double slashes."""
        return v.rstrip("/")

    @property
    def pdf_replacement_enabled(self) -> bool:
        """Whether archive PDF replacement is available."""
        return self.archive_dir is not None and (self.database_url is not None)


# ---------------------------------------------------------------------------
# Tags Configuration
# ---------------------------------------------------------------------------


class TagsConfig(ConfigBaseModel):
    """Tag configuration for document workflow.

    Documents are selected for processing based on tag filters.
    Include tags select documents; exclude tags filter them out.

    Attributes:
        prefix: Prefix for workflow tags (e.g., "smart-ocr" creates
            tags like "smart-ocr:pending").
        include: Documents with ANY of these tags are processed.
        exclude: Documents with ANY of these tags are skipped.
    """

    prefix: str = Field(
        default="smart-ocr",
        description="Prefix for workflow tags",
    )
    include: list[str] = Field(
        default_factory=lambda: ["smart-ocr:pending"],
        description="Include documents with ANY of these tags",
    )
    exclude: list[str] = Field(
        default_factory=lambda: [
            "smart-ocr:completed",
            "smart-ocr:failed",
            "smart-ocr:skip",
        ],
        description="Exclude documents with ANY of these tags",
    )


# ---------------------------------------------------------------------------
# Pipeline - Stage 1 (OCR)
# ---------------------------------------------------------------------------


class OCRmyPDFConfig(ConfigBaseModel):
    """OCRmyPDF configuration options.

    These settings are passed to OCRmyPDF when processing documents.

    Attributes:
        deskew: Deskew pages before OCR to correct rotation.
        clean: Clean/despeckle pages before OCR.
        rotate_pages: Auto-rotate pages to correct orientation.
        language: Tesseract language code(s), e.g., "eng" or "eng+fra".
        extra_args: Additional OCRmyPDF command-line arguments.
    """

    deskew: bool = Field(default=True, description="Deskew pages before OCR")
    clean: bool = Field(default=True, description="Clean pages before OCR")
    rotate_pages: bool = Field(default=True, description="Auto-rotate pages")
    language: str = Field(
        default="eng",
        description="Tesseract language code(s)",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional OCRmyPDF command-line arguments",
    )


class LayoutDetectionConfig(ConfigBaseModel):
    """Surya layout detection configuration.

    Layout detection identifies text regions, tables, figures, and other
    elements in the document. This improves OCR accuracy for complex layouts.

    Attributes:
        enabled: Whether to use layout detection before OCR.
        confidence_threshold: Minimum confidence for detected regions (0.0-1.0).
        exclude_regions: Region types to exclude from OCR (e.g., headers, footers).
    """

    enabled: bool = Field(default=True, description="Enable layout detection")
    confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence for detected regions",
        ),
    ] = 0.5
    exclude_regions: list[str] = Field(
        default_factory=lambda: [
            "advertisement",
            "page_number",
            "header",
            "footer",
        ],
        description="Region types to exclude from OCR",
    )


class Stage1Config(ConfigBaseModel):
    """Stage 1 (OCR) configuration.

    Stage 1 adds a searchable text layer to scanned PDFs using OCRmyPDF
    with optional Surya layout detection for complex documents.

    Attributes:
        enabled: Whether Stage 1 processing is enabled.
        born_digital_handling: How to handle documents with existing text.
        ocrmypdf: OCRmyPDF-specific options.
        layout_detection: Surya layout detection options.
    """

    enabled: bool = Field(default=True, description="Enable Stage 1 processing")
    born_digital_handling: BornDigitalHandling = Field(
        default=BornDigitalHandling.SKIP,
        description="How to handle documents with existing text",
    )
    ocrmypdf: OCRmyPDFConfig = Field(default_factory=OCRmyPDFConfig)
    layout_detection: LayoutDetectionConfig = Field(
        default_factory=LayoutDetectionConfig,
    )


# ---------------------------------------------------------------------------
# Pipeline - Stage 2 (Markdown)
# ---------------------------------------------------------------------------


class LLMConfig(ConfigBaseModel):
    """LLM configuration for Marker (optional).

    When use_llm is enabled in MarkerConfig, these settings configure
    the LLM provider. Note: LLM usage may introduce hallucinations.

    Attributes:
        provider: LLM provider (openai, anthropic, ollama).
        model: Model identifier (e.g., "granite3.2-vision:2b", "gpt-4o-mini").
        api_key: API key (supports ${VAR} interpolation).
    """

    provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    model: str = Field(default="granite3.2-vision:2b")
    api_key: str | None = Field(
        default=None,
        description="API key (supports ${VAR} interpolation)",
    )


class MarkerConfig(ConfigBaseModel):
    """Marker PDF-to-Markdown configuration.

    Marker extracts structured content from PDFs and converts it to
    Markdown, preserving headings, lists, tables, and code blocks.

    Attributes:
        use_llm: Enable LLM assistance (may introduce hallucinations).
        llm: LLM configuration (only used if use_llm is True).
    """

    use_llm: bool = Field(
        default=False,
        description="Enable LLM assistance (may introduce hallucinations)",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)


class Stage2Config(ConfigBaseModel):
    """Stage 2 (Markdown conversion) configuration.

    Stage 2 converts PDFs to Markdown using Marker and updates the
    document's content field in Paperless-ngx.

    Attributes:
        enabled: Whether Stage 2 processing is enabled.
        content_mode: How to handle existing content (replace or append).
        marker: Marker-specific options.
    """

    enabled: bool = Field(default=True, description="Enable Stage 2 processing")
    content_mode: ContentMode = Field(
        default=ContentMode.REPLACE,
        description="How to handle existing content",
    )
    marker: MarkerConfig = Field(default_factory=MarkerConfig)


class PipelineConfig(ConfigBaseModel):
    """Processing pipeline configuration.

    The pipeline consists of two stages that can be independently enabled:
    - Stage 1: OCR (adds searchable text layer)
    - Stage 2: Markdown (converts to structured Markdown)

    Attributes:
        stage1: Stage 1 (OCR) configuration.
        stage2: Stage 2 (Markdown) configuration.
    """

    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)


# ---------------------------------------------------------------------------
# Integration Modes
# ---------------------------------------------------------------------------


class PollingConfig(ConfigBaseModel):
    """Polling integration configuration.

    When enabled, the service periodically polls Paperless-ngx for
    documents matching the tag filters.

    Attributes:
        enabled: Whether polling is enabled.
        interval_seconds: Time between polls (10 seconds to 24 hours).
        batch_size: Maximum documents to process per poll cycle.
    """

    enabled: bool = Field(default=True)
    interval_seconds: Annotated[
        int,
        Field(
            ge=10,
            le=86400,
            description="Polling interval (10s to 24h)",
        ),
    ] = 300
    batch_size: Annotated[
        int,
        Field(
            ge=1,
            le=100,
            description="Max documents per poll cycle",
        ),
    ] = 10


class WebhookConfig(ConfigBaseModel):
    """Webhook integration configuration.

    When enabled, the service accepts webhooks from Paperless-ngx
    Workflows at the /api/webhook endpoint.

    Attributes:
        enabled: Whether webhook receiver is enabled.
        secret: Webhook secret for request validation (supports ${VAR}).
    """

    enabled: bool = Field(default=False)
    secret: str | None = Field(
        default=None,
        description="Webhook secret for validation (supports ${VAR})",
    )


class PostConsumeConfig(ConfigBaseModel):
    """Post-consume script integration configuration.

    When enabled, the service can run in CLI mode triggered by
    Paperless-ngx's post-consume script feature.

    Attributes:
        enabled: Whether post-consume mode is enabled.
    """

    enabled: bool = Field(default=False)


class IntegrationConfig(ConfigBaseModel):
    """Integration mode configuration.

    Multiple integration modes can be enabled simultaneously.

    Attributes:
        polling: Polling integration settings.
        webhook: Webhook integration settings.
        post_consume: Post-consume script settings.
    """

    polling: PollingConfig = Field(default_factory=PollingConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    post_consume: PostConsumeConfig = Field(default_factory=PostConsumeConfig)


# ---------------------------------------------------------------------------
# Other Settings
# ---------------------------------------------------------------------------


class AutoProcessingConfig(ConfigBaseModel):
    """Auto-processing configuration.

    Controls whether documents matching tag filters are automatically
    processed. Manual processing via the web UI is always available.

    Attributes:
        enabled: Whether automatic processing is enabled.
    """

    enabled: bool = Field(default=True)


class JobsConfig(ConfigBaseModel):
    """Background job processing configuration.

    Controls the concurrent processing capacity and timeout settings.

    Attributes:
        workers: Number of concurrent worker tasks.
        timeout: Maximum processing time per document in seconds.
    """

    workers: Annotated[
        int,
        Field(ge=1, le=32, description="Number of concurrent workers"),
    ] = 2
    timeout: Annotated[
        int,
        Field(
            ge=60,
            le=7200,
            description="Timeout per document in seconds",
        ),
    ] = 600


class GPUConfig(ConfigBaseModel):
    """GPU acceleration configuration.

    Controls GPU usage for Surya layout detection and Marker.

    Attributes:
        enabled: GPU mode (auto, cuda, or cpu).
    """

    enabled: GPUMode = Field(default=GPUMode.AUTO)


class WebConfig(ConfigBaseModel):
    """Web UI and API server configuration.

    Attributes:
        host: Bind address for the web server.
        port: Port number for the web server.
        theme: UI theme mode (auto, dark, or light).
    """

    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: Annotated[
        int,
        Field(ge=1, le=65535, description="Port number"),
    ] = 8080
    theme: ThemeMode = Field(default=ThemeMode.AUTO)
    cookie_secure: bool = Field(
        default=False,
        description="Set Secure flag on auth cookie (requires HTTPS)",
    )


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class LoggingConfig(ConfigBaseModel):
    """Logging configuration.

    Attributes:
        level: Log verbosity level.
        format: Log output format (json or console).
    """

    level: LogLevel = Field(default=LogLevel.INFO)
    format: LogFormat = Field(default=LogFormat.JSON)


class MetricsConfig(ConfigBaseModel):
    """Prometheus metrics configuration.

    Attributes:
        enabled: Whether metrics endpoint is enabled.
        port: Port for the metrics endpoint.
    """

    enabled: bool = Field(default=True)
    port: Annotated[
        int,
        Field(ge=1, le=65535, description="Metrics port"),
    ] = 9090


class TracingConfig(ConfigBaseModel):
    """OpenTelemetry tracing configuration.

    Attributes:
        enabled: Whether tracing is enabled.
        otlp_endpoint: OTLP exporter endpoint URL.
    """

    enabled: bool = Field(default=False)
    otlp_endpoint: str = Field(default="http://localhost:4317")


class ObservabilityConfig(ConfigBaseModel):
    """Observability configuration.

    Groups logging, metrics, and tracing settings.

    Attributes:
        logging: Logging configuration.
        metrics: Prometheus metrics configuration.
        tracing: OpenTelemetry tracing configuration.
    """

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
