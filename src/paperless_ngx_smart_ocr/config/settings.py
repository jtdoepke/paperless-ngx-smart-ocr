"""Settings management for paperless-ngx-smart-ocr.

This module provides the main Settings class and functions for loading
configuration from YAML files and environment variables.

Example:
    >>> from paperless_ngx_smart_ocr.config import load_settings
    >>> settings = load_settings()
    >>> print(settings.paperless.url)
    http://localhost:8000
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from paperless_ngx_smart_ocr.config.exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
)
from paperless_ngx_smart_ocr.config.schema import (
    AutoProcessingConfig,
    GPUConfig,
    IntegrationConfig,
    JobsConfig,
    ObservabilityConfig,
    PaperlessConfig,
    PipelineConfig,
    TagsConfig,
    WebConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "Settings",
    "clear_settings_cache",
    "find_config_file",
    "get_settings",
    "load_settings",
]


# ---------------------------------------------------------------------------
# Environment Variable Interpolation
# ---------------------------------------------------------------------------

# Pattern for ${VAR} and ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env_vars(value: object) -> object:
    """Recursively interpolate ${VAR} and ${VAR:-default} in strings.

    Args:
        value: Value to interpolate (string, dict, list, or other).

    Returns:
        Value with environment variables interpolated. Non-string values
        are returned unchanged, except dicts and lists which are processed
        recursively.

    Example:
        >>> os.environ["MY_TOKEN"] = "secret123"
        >>> _interpolate_env_vars("Bearer ${MY_TOKEN}")
        'Bearer secret123'
        >>> _interpolate_env_vars("${MISSING:-default_value}")
        'default_value'
    """
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)  # May be None
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return ""

        return _ENV_VAR_PATTERN.sub(replace, value)

    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


# ---------------------------------------------------------------------------
# Custom YAML Settings Source with Interpolation
# ---------------------------------------------------------------------------


class _InterpolatingYamlConfigSettingsSource(YamlConfigSettingsSource):
    """YAML settings source with ${VAR} interpolation support.

    Extends the standard YamlConfigSettingsSource to perform environment
    variable interpolation on loaded YAML values before passing them to
    Pydantic for validation.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_file: Path | str | None = None,
    ) -> None:
        """Initialize the YAML settings source.

        Args:
            settings_cls: The Settings class being configured.
            yaml_file: Path to YAML file, or None to use model_config setting.
        """
        # Only pass yaml_file if explicitly provided, otherwise let parent
        # use the value from model_config['yaml_file']
        if yaml_file is not None:
            super().__init__(settings_cls, yaml_file=yaml_file)
        else:
            super().__init__(settings_cls)

    def _read_files(
        self,
        files: Path | str | Sequence[Path | str] | None,
    ) -> dict[str, Any]:
        """Read and parse YAML files with environment variable interpolation.

        Args:
            files: Path(s) to YAML file(s) to read.

        Returns:
            Dictionary of parsed YAML with interpolated environment variables.
        """
        raw_data = super()._read_files(files)
        interpolated = _interpolate_env_vars(raw_data)
        # Type narrowing: _interpolate_env_vars returns dict for dict input
        if not isinstance(interpolated, dict):  # pragma: no cover
            return {}
        return interpolated


# ---------------------------------------------------------------------------
# Settings Class
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from YAML file and environment variables.

    Settings are loaded in priority order (highest to lowest):
    1. Constructor arguments
    2. Environment variables (SMARTOCR_* and PAPERLESS_*)
    3. YAML configuration file
    4. Default values

    The YAML file supports ${VAR} and ${VAR:-default} syntax for
    environment variable interpolation.

    Attributes:
        paperless: Paperless-ngx connection settings.
        tags: Tag-based workflow settings.
        pipeline: Processing pipeline settings.
        integration: Integration mode settings.
        auto_processing: Auto-processing settings.
        jobs: Background job settings.
        gpu: GPU acceleration settings.
        web: Web UI settings.
        observability: Logging, metrics, and tracing settings.

    Example:
        >>> # Load from default locations
        >>> settings = load_settings()
        >>> print(settings.paperless.url)
        http://localhost:8000

        >>> # Load from specific file
        >>> settings = load_settings(Path("/etc/smart-ocr/config.yaml"))

        >>> # Access nested settings
        >>> print(settings.pipeline.stage1.enabled)
        True
    """

    model_config = SettingsConfigDict(
        # YAML file configuration (default, can be overridden via _yaml_file_override)
        yaml_file=None,  # No default file - search paths used instead
        yaml_file_encoding="utf-8",
        # Environment variable configuration
        env_prefix="SMARTOCR_",
        env_nested_delimiter="__",
        # Extra fields handling
        extra="ignore",
        # Validation
        validate_default=True,
    )

    # Search paths for config file (class variable, not a setting)
    CONFIG_SEARCH_PATHS: ClassVar[list[Path]] = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".config" / "smart-ocr" / "config.yaml",
        Path("/etc/smart-ocr/config.yaml"),
    ]

    # Override for yaml_file path (set by load_settings before instantiation)
    _yaml_file_override: ClassVar[Path | str | None] = None

    # Top-level configuration sections
    paperless: PaperlessConfig = PaperlessConfig()
    tags: TagsConfig = TagsConfig()
    pipeline: PipelineConfig = PipelineConfig()
    integration: IntegrationConfig = IntegrationConfig()
    auto_processing: AutoProcessingConfig = AutoProcessingConfig()
    jobs: JobsConfig = JobsConfig()
    gpu: GPUConfig = GPUConfig()
    web: WebConfig = WebConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @model_validator(mode="after")
    def resolve_paperless_token(self) -> Settings:
        """Resolve Paperless token from various sources.

        Token resolution order:
        1. Direct token value (if set)
        2. Token file (if token_file is set and file exists)
        3. PAPERLESS_TOKEN environment variable

        Returns:
            Self with resolved token.

        Raises:
            ValueError: If token_file is specified but file doesn't exist.
        """
        # If token is already set, use it
        if self.paperless.token:
            return self

        # Try to read from token_file
        if self.paperless.token_file:
            token_path = self.paperless.token_file
            if token_path.is_file():
                # Use object.__setattr__ since model may be frozen
                object.__setattr__(
                    self.paperless,
                    "token",
                    token_path.read_text().strip(),
                )
            else:
                msg = f"Token file not found: {token_path}"
                raise ValueError(msg)
            return self

        # Fall back to PAPERLESS_TOKEN environment variable
        env_token = os.environ.get("PAPERLESS_TOKEN")
        if env_token:
            object.__setattr__(self.paperless, "token", env_token)

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings source priority.

        Priority order (highest to lowest):
        1. Init settings (constructor arguments)
        2. Environment variables
        3. YAML configuration file (with interpolation)
        4. File secrets

        Note: dotenv is excluded as we use YAML files instead.

        Args:
            settings_cls: The Settings class being configured.
            init_settings: Source for constructor arguments.
            env_settings: Source for environment variables.
            dotenv_settings: Source for .env files (unused).
            file_secret_settings: Source for Docker secrets.

        Returns:
            Tuple of settings sources in priority order.
        """
        # Use the override path if set, otherwise None (uses model_config)
        yaml_file = cls._yaml_file_override
        return (
            init_settings,
            env_settings,
            _InterpolatingYamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
            file_secret_settings,
        )


# ---------------------------------------------------------------------------
# Settings Loading Functions
# ---------------------------------------------------------------------------

_cached_settings: Settings | None = None


def find_config_file(config_path: Path | str | None = None) -> Path | None:
    """Find the configuration file.

    Args:
        config_path: Explicit path to config file, or None to search
            default locations.

    Returns:
        Path to config file if found, None otherwise.

    Example:
        >>> find_config_file()  # Searches default paths
        PosixPath('config.yaml')
        >>> find_config_file("/custom/path/config.yaml")
        PosixPath('/custom/path/config.yaml')
    """
    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            return path
        return None

    # Search standard locations
    for search_path in Settings.CONFIG_SEARCH_PATHS:
        if search_path.is_file():
            return search_path

    return None


def load_settings(
    config_path: Path | str | None = None,
    *,
    require_config_file: bool = False,
) -> Settings:
    """Load and validate application settings.

    Loads settings from YAML file and/or environment variables. The loaded
    settings are cached for subsequent calls to get_settings().

    Args:
        config_path: Path to YAML config file. If None, searches standard
            locations (./config.yaml, ./config.yml,
            ~/.config/smart-ocr/config.yaml, /etc/smart-ocr/config.yaml).
        require_config_file: If True, raise error when no config file found.
            If False (default), proceed with environment variables and defaults.

    Returns:
        Validated Settings instance.

    Raises:
        ConfigurationFileNotFoundError: When require_config_file=True and
            no config file is found.
        ConfigurationValidationError: When configuration validation fails.

    Example:
        >>> # Load from default locations or environment
        >>> settings = load_settings()

        >>> # Load from specific file
        >>> settings = load_settings("/path/to/config.yaml")

        >>> # Require a config file
        >>> settings = load_settings(require_config_file=True)
    """
    global _cached_settings  # noqa: PLW0603

    config_file = find_config_file(config_path)

    if config_file is None and require_config_file:
        raise ConfigurationFileNotFoundError(
            path=str(config_path) if config_path else None,
            searched_paths=[str(p) for p in Settings.CONFIG_SEARCH_PATHS],
        )

    try:
        # Set the yaml_file override before creating Settings instance
        # This is read by settings_customise_sources()
        Settings._yaml_file_override = config_file  # noqa: SLF001

        try:
            settings = Settings()
        finally:
            # Reset override to avoid affecting future calls
            Settings._yaml_file_override = None  # noqa: SLF001

    except ConfigurationError:
        # Re-raise our own exceptions
        raise
    except Exception as exc:
        msg = f"Failed to load configuration: {exc}"
        raise ConfigurationValidationError(msg) from exc
    else:
        _cached_settings = settings
        return settings


def get_settings() -> Settings:
    """Get the cached settings instance, loading if necessary.

    This function provides access to the singleton settings instance.
    If settings have not been loaded yet, they will be loaded with
    default options (searching standard config file locations).

    Returns:
        The cached Settings instance.

    Example:
        >>> settings = get_settings()
        >>> print(settings.web.port)
        8080
    """
    global _cached_settings  # noqa: PLW0603

    if _cached_settings is None:
        _cached_settings = load_settings()

    return _cached_settings


def clear_settings_cache() -> None:
    """Clear the cached settings instance.

    This function is primarily useful for testing, allowing tests to
    start with a fresh settings state.

    Example:
        >>> clear_settings_cache()
        >>> # Next call to get_settings() will reload
    """
    global _cached_settings  # noqa: PLW0603
    _cached_settings = None
