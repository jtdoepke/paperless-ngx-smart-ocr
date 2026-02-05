"""Configuration module for paperless-ngx-smart-ocr.

This module provides configuration management using Pydantic settings with
support for YAML files and environment variable overrides. Configuration
values support ${VAR} and ${VAR:-default} syntax for environment variable
interpolation.

Example:
    >>> from paperless_ngx_smart_ocr.config import load_settings, get_settings
    >>>
    >>> # Load settings from file or environment
    >>> settings = load_settings()
    >>>
    >>> # Access configuration
    >>> print(settings.paperless.url)
    http://localhost:8000
    >>> print(settings.pipeline.stage1.enabled)
    True
    >>>
    >>> # Use cached singleton
    >>> settings = get_settings()
"""

from __future__ import annotations

from paperless_ngx_smart_ocr.config.exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
)
from paperless_ngx_smart_ocr.config.schema import (
    AutoProcessingConfig,
    BornDigitalHandling,
    ConfigBaseModel,
    ContentMode,
    GPUConfig,
    GPUMode,
    IntegrationConfig,
    JobsConfig,
    LayoutDetectionConfig,
    LLMConfig,
    LLMProvider,
    LogFormat,
    LoggingConfig,
    LogLevel,
    MarkerConfig,
    MetricsConfig,
    ObservabilityConfig,
    OCRmyPDFConfig,
    PaperlessConfig,
    PipelineConfig,
    PollingConfig,
    PostConsumeConfig,
    Stage1Config,
    Stage2Config,
    TagsConfig,
    ThemeMode,
    TracingConfig,
    WebConfig,
    WebhookConfig,
)
from paperless_ngx_smart_ocr.config.settings import (
    Settings,
    clear_settings_cache,
    find_config_file,
    get_settings,
    load_settings,
)


__all__ = [
    # Configuration section models (for type hints)
    "AutoProcessingConfig",
    # Enums
    "BornDigitalHandling",
    # Base model
    "ConfigBaseModel",
    # Exceptions
    "ConfigurationError",
    "ConfigurationFileNotFoundError",
    "ConfigurationValidationError",
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
    # Main settings class and functions
    "Settings",
    "Stage1Config",
    "Stage2Config",
    "TagsConfig",
    "ThemeMode",
    "TracingConfig",
    "WebConfig",
    "WebhookConfig",
    "clear_settings_cache",
    "find_config_file",
    "get_settings",
    "load_settings",
]
