"""Unit tests for the configuration module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from paperless_ngx_smart_ocr.config import (
    BornDigitalHandling,
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
    ContentMode,
    GPUMode,
    LayoutDetectionConfig,
    LLMProvider,
    LogFormat,
    LogLevel,
    OCRmyPDFConfig,
    PaperlessConfig,
    PipelineConfig,
    PollingConfig,
    Settings,
    Stage1Config,
    Stage2Config,
    TagsConfig,
    ThemeMode,
    WebConfig,
    clear_settings_cache,
    find_config_file,
    get_settings,
    load_settings,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_settings() -> Generator[None, None, None]:
    """Clear settings cache before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """Create a sample configuration file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
paperless:
  url: "http://test-paperless:8000"
  token: "test-token-12345"

tags:
  prefix: "test-ocr"
  include:
    - "test-ocr:pending"
  exclude:
    - "test-ocr:done"

pipeline:
  stage1:
    enabled: true
    born_digital_handling: "force"
  stage2:
    enabled: false
    content_mode: "append"

web:
  host: "127.0.0.1"
  port: 9000
  theme: "dark"
""")
    return config_file


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Create a minimal configuration file (uses all defaults)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("# Empty config - uses all defaults\n")
    return config_file


@pytest.fixture
def config_with_interpolation(tmp_path: Path) -> Path:
    """Create a config file with environment variable interpolation."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
paperless:
  url: "${TEST_PAPERLESS_URL:-http://default:8000}"
  token: "${TEST_PAPERLESS_TOKEN}"

integration:
  webhook:
    enabled: true
    secret: "${TEST_WEBHOOK_SECRET:-default-secret}"
""")
    return config_file


@pytest.fixture
def config_with_token_file(tmp_path: Path) -> tuple[Path, Path]:
    """Create a config file that references a token file."""
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token-from-file\n")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"""
paperless:
  url: "http://paperless:8000"
  token_file: "{token_file}"
""")
    return config_file, token_file


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for configuration enums."""

    def test_born_digital_handling_values(self) -> None:
        """Test BornDigitalHandling enum values."""
        assert BornDigitalHandling.SKIP == "skip"
        assert BornDigitalHandling.FORCE == "force"

    def test_content_mode_values(self) -> None:
        """Test ContentMode enum values."""
        assert ContentMode.REPLACE == "replace"
        assert ContentMode.APPEND == "append"

    def test_gpu_mode_values(self) -> None:
        """Test GPUMode enum values."""
        assert GPUMode.AUTO == "auto"
        assert GPUMode.CUDA == "cuda"
        assert GPUMode.CPU == "cpu"

    def test_log_format_values(self) -> None:
        """Test LogFormat enum values."""
        assert LogFormat.JSON == "json"
        assert LogFormat.CONSOLE == "console"

    def test_log_level_values(self) -> None:
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_theme_mode_values(self) -> None:
        """Test ThemeMode enum values."""
        assert ThemeMode.AUTO == "auto"
        assert ThemeMode.DARK == "dark"
        assert ThemeMode.LIGHT == "light"

    def test_llm_provider_values(self) -> None:
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.OLLAMA == "ollama"


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for configuration schema validation."""

    def test_paperless_config_defaults(self) -> None:
        """Test PaperlessConfig default values."""
        config = PaperlessConfig()
        assert config.url == "http://localhost:8000"
        assert config.token is None
        assert config.token_file is None

    def test_paperless_config_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from URL."""
        config = PaperlessConfig(url="http://example.com/api/")
        assert config.url == "http://example.com/api"

    def test_tags_config_defaults(self) -> None:
        """Test TagsConfig default values."""
        config = TagsConfig()
        assert config.prefix == "smart-ocr"
        assert config.include == ["smart-ocr:pending"]
        assert "smart-ocr:completed" in config.exclude
        assert "smart-ocr:failed" in config.exclude
        assert "smart-ocr:skip" in config.exclude

    def test_pipeline_config_defaults(self) -> None:
        """Test PipelineConfig default values."""
        config = PipelineConfig()
        assert config.stage1.enabled is True
        assert config.stage1.born_digital_handling == BornDigitalHandling.SKIP
        assert config.stage2.enabled is True
        assert config.stage2.content_mode == ContentMode.REPLACE

    def test_ocrmypdf_config_defaults(self) -> None:
        """Test OCRmyPDFConfig default values."""
        config = OCRmyPDFConfig()
        assert config.deskew is True
        assert config.clean is True
        assert config.rotate_pages is True
        assert config.language == "eng"
        assert config.extra_args == []

    def test_layout_detection_config_defaults(self) -> None:
        """Test LayoutDetectionConfig default values."""
        config = LayoutDetectionConfig()
        assert config.enabled is True
        assert config.confidence_threshold == 0.5
        assert "header" in config.exclude_regions
        assert "footer" in config.exclude_regions

    def test_layout_detection_confidence_threshold_bounds(self) -> None:
        """Test confidence_threshold validation bounds."""
        # Valid values
        LayoutDetectionConfig(confidence_threshold=0.0)
        LayoutDetectionConfig(confidence_threshold=0.5)
        LayoutDetectionConfig(confidence_threshold=1.0)

        # Invalid: below 0
        with pytest.raises(ValidationError):
            LayoutDetectionConfig(confidence_threshold=-0.1)

        # Invalid: above 1
        with pytest.raises(ValidationError):
            LayoutDetectionConfig(confidence_threshold=1.1)

    def test_polling_config_interval_bounds(self) -> None:
        """Test polling interval validation bounds."""
        # Valid values
        PollingConfig(interval_seconds=10)
        PollingConfig(interval_seconds=86400)

        # Invalid: below 10
        with pytest.raises(ValidationError):
            PollingConfig(interval_seconds=5)

        # Invalid: above 86400 (24 hours)
        with pytest.raises(ValidationError):
            PollingConfig(interval_seconds=90000)

    def test_polling_config_batch_size_bounds(self) -> None:
        """Test polling batch_size validation bounds."""
        # Valid values
        PollingConfig(batch_size=1)
        PollingConfig(batch_size=100)

        # Invalid: below 1
        with pytest.raises(ValidationError):
            PollingConfig(batch_size=0)

        # Invalid: above 100
        with pytest.raises(ValidationError):
            PollingConfig(batch_size=101)

    def test_web_config_port_bounds(self) -> None:
        """Test web port validation bounds."""
        # Valid values
        WebConfig(port=1)
        WebConfig(port=8080)
        WebConfig(port=65535)

        # Invalid: below 1
        with pytest.raises(ValidationError):
            WebConfig(port=0)

        # Invalid: above 65535
        with pytest.raises(ValidationError):
            WebConfig(port=70000)

    def test_stage1_config_enum_validation(self) -> None:
        """Test Stage1Config enum validation."""
        # Valid enum value
        config = Stage1Config(born_digital_handling="force")
        assert config.born_digital_handling == BornDigitalHandling.FORCE

        # Invalid enum value
        with pytest.raises(ValidationError):
            Stage1Config(born_digital_handling="invalid")

    def test_stage2_config_enum_validation(self) -> None:
        """Test Stage2Config enum validation."""
        # Valid enum value
        config = Stage2Config(content_mode="append")
        assert config.content_mode == ContentMode.APPEND

        # Invalid enum value
        with pytest.raises(ValidationError):
            Stage2Config(content_mode="invalid")

    def test_config_forbids_extra_fields(self) -> None:
        """Test that unknown fields raise validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            PaperlessConfig(url="http://test", unknown_field="value")

        assert "extra_forbidden" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Settings Loading Tests
# ---------------------------------------------------------------------------


class TestSettingsLoading:
    """Tests for settings loading functionality."""

    def test_load_settings_with_defaults(self) -> None:
        """Test loading settings with all defaults (no config file)."""
        settings = load_settings()

        assert settings.paperless.url == "http://localhost:8000"
        assert settings.pipeline.stage1.enabled is True
        assert settings.web.port == 8080

    def test_load_settings_from_yaml(self, sample_config_yaml: Path) -> None:
        """Test loading settings from a YAML file."""
        settings = load_settings(sample_config_yaml)

        assert settings.paperless.url == "http://test-paperless:8000"
        assert settings.paperless.token == "test-token-12345"  # noqa: S105
        assert settings.tags.prefix == "test-ocr"
        born_digital = settings.pipeline.stage1.born_digital_handling
        assert born_digital == BornDigitalHandling.FORCE
        assert settings.pipeline.stage2.enabled is False
        assert settings.pipeline.stage2.content_mode == ContentMode.APPEND
        assert settings.web.host == "127.0.0.1"
        assert settings.web.port == 9000
        assert settings.web.theme == ThemeMode.DARK

    def test_load_settings_minimal_yaml(self, minimal_config_yaml: Path) -> None:
        """Test loading settings from minimal YAML (uses defaults)."""
        settings = load_settings(minimal_config_yaml)

        # Should use all default values
        assert settings.paperless.url == "http://localhost:8000"
        assert settings.pipeline.stage1.enabled is True
        assert settings.web.port == 8080

    def test_load_settings_requires_file(self) -> None:
        """Test that require_config_file=True raises on missing file."""
        with pytest.raises(ConfigurationFileNotFoundError) as exc_info:
            load_settings("/nonexistent/config.yaml", require_config_file=True)

        assert "not found" in exc_info.value.message.lower()

    def test_load_settings_nonexistent_file_optional(self) -> None:
        """Test that missing file is OK when not required."""
        # Should not raise, uses defaults
        settings = load_settings("/nonexistent/config.yaml")
        assert settings.paperless.url == "http://localhost:8000"


class TestEnvironmentVariableInterpolation:
    """Tests for ${VAR} interpolation in YAML values."""

    def test_interpolation_with_value(
        self,
        config_with_interpolation: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test interpolation when environment variable is set."""
        monkeypatch.setenv("TEST_PAPERLESS_URL", "http://env-paperless:9000")
        monkeypatch.setenv("TEST_PAPERLESS_TOKEN", "env-token-xyz")

        settings = load_settings(config_with_interpolation)

        assert settings.paperless.url == "http://env-paperless:9000"
        assert settings.paperless.token == "env-token-xyz"  # noqa: S105

    def test_interpolation_with_default(
        self,
        config_with_interpolation: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test interpolation uses default when variable not set."""
        # Only set TOKEN, not URL (which has a default)
        monkeypatch.setenv("TEST_PAPERLESS_TOKEN", "env-token-xyz")
        # Ensure URL is not set
        monkeypatch.delenv("TEST_PAPERLESS_URL", raising=False)

        settings = load_settings(config_with_interpolation)

        assert settings.paperless.url == "http://default:8000"
        assert settings.paperless.token == "env-token-xyz"  # noqa: S105

    def test_interpolation_empty_when_missing(
        self,
        config_with_interpolation: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test interpolation returns empty string for missing var without default."""
        monkeypatch.delenv("TEST_PAPERLESS_TOKEN", raising=False)
        monkeypatch.delenv("TEST_PAPERLESS_URL", raising=False)

        settings = load_settings(config_with_interpolation)

        # TOKEN has no default, so becomes empty string
        assert settings.paperless.token == ""
        # URL has default
        assert settings.paperless.url == "http://default:8000"


class TestTokenFileResolution:
    """Tests for token_file resolution."""

    def test_token_from_file(
        self,
        config_with_token_file: tuple[Path, Path],
    ) -> None:
        """Test token is read from token_file."""
        config_file, _ = config_with_token_file
        settings = load_settings(config_file)

        assert settings.paperless.token == "secret-token-from-file"  # noqa: S105

    def test_token_file_not_found(self, tmp_path: Path) -> None:
        """Test error when token_file doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
paperless:
  token_file: "/nonexistent/token.txt"
""")

        with pytest.raises(ConfigurationValidationError) as exc_info:
            load_settings(config_file)

        assert "token file not found" in exc_info.value.message.lower()

    def test_token_takes_precedence_over_file(self, tmp_path: Path) -> None:
        """Test direct token takes precedence over token_file."""
        token_file = tmp_path / "token.txt"
        token_file.write_text("file-token")

        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
paperless:
  token: "direct-token"
  token_file: "{token_file}"
""")

        settings = load_settings(config_file)
        assert settings.paperless.token == "direct-token"  # noqa: S105

    def test_paperless_token_env_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test PAPERLESS_TOKEN environment variable fallback."""
        monkeypatch.setenv("PAPERLESS_TOKEN", "env-fallback-token")

        settings = load_settings()
        assert settings.paperless.token == "env-fallback-token"  # noqa: S105


class TestEnvironmentVariableOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_simple(
        self,
        sample_config_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test environment variable overrides YAML value."""
        monkeypatch.setenv("SMARTOCR_WEB__PORT", "3000")

        settings = load_settings(sample_config_yaml)

        # YAML has port=9000, env should override to 3000
        assert settings.web.port == 3000

    def test_env_override_nested(
        self,
        minimal_config_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test nested environment variable override."""
        monkeypatch.setenv("SMARTOCR_PIPELINE__STAGE1__ENABLED", "false")

        settings = load_settings(minimal_config_yaml)

        assert settings.pipeline.stage1.enabled is False

    def test_env_override_deeply_nested(
        self,
        minimal_config_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test deeply nested environment variable override."""
        monkeypatch.setenv("SMARTOCR_PIPELINE__STAGE2__MARKER__USE_LLM", "true")

        settings = load_settings(minimal_config_yaml)

        assert settings.pipeline.stage2.marker.use_llm is True


# ---------------------------------------------------------------------------
# Settings Caching Tests
# ---------------------------------------------------------------------------


class TestSettingsCaching:
    """Tests for settings caching behavior."""

    def test_get_settings_caches(self) -> None:
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_clear_settings_cache(self) -> None:
        """Test that clear_settings_cache clears the cache."""
        settings1 = get_settings()
        clear_settings_cache()
        settings2 = get_settings()

        # Should be different instances (both valid, but not same object)
        assert settings1 is not settings2

    def test_load_settings_updates_cache(self, sample_config_yaml: Path) -> None:
        """Test that load_settings updates the cache."""
        settings1 = get_settings()
        settings2 = load_settings(sample_config_yaml)
        settings3 = get_settings()

        # load_settings should update the cache
        assert settings3 is settings2
        assert settings3 is not settings1


# ---------------------------------------------------------------------------
# Find Config File Tests
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_find_explicit_path(self, sample_config_yaml: Path) -> None:
        """Test finding config at explicit path."""
        result = find_config_file(sample_config_yaml)
        assert result == sample_config_yaml

    def test_find_explicit_path_string(self, sample_config_yaml: Path) -> None:
        """Test finding config at explicit path as string."""
        result = find_config_file(str(sample_config_yaml))
        assert result == sample_config_yaml

    def test_find_nonexistent_returns_none(self) -> None:
        """Test that nonexistent path returns None."""
        result = find_config_file("/nonexistent/config.yaml")
        assert result is None

    def test_find_searches_default_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that None searches default paths."""
        # Create config.yaml in current directory
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("web:\n  port: 1234\n")

        result = find_config_file(None)
        # Result is relative path from CONFIG_SEARCH_PATHS
        assert result is not None
        assert result.resolve() == config_file.resolve()


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for configuration exceptions."""

    def test_configuration_error_base(self) -> None:
        """Test ConfigurationError base exception."""
        exc = ConfigurationError("Test error message")
        assert exc.message == "Test error message"
        assert str(exc) == "Test error message"

    def test_configuration_file_not_found_error(self) -> None:
        """Test ConfigurationFileNotFoundError."""
        exc = ConfigurationFileNotFoundError(
            path="/path/to/config.yaml",
            searched_paths=["/a", "/b", "/c"],
        )
        assert exc.path == "/path/to/config.yaml"
        assert exc.searched_paths == ["/a", "/b", "/c"]
        assert "/a" in exc.message
        assert "/b" in exc.message
        assert "/c" in exc.message

    def test_configuration_file_not_found_error_no_paths(self) -> None:
        """Test ConfigurationFileNotFoundError with just path."""
        exc = ConfigurationFileNotFoundError(path="/path/to/config.yaml")
        assert "/path/to/config.yaml" in exc.message

    def test_configuration_validation_error(self) -> None:
        """Test ConfigurationValidationError."""
        errors = [{"loc": ["field"], "msg": "invalid"}]
        exc = ConfigurationValidationError("Validation failed", errors=errors)
        assert exc.message == "Validation failed"
        assert exc.errors == errors

    def test_configuration_validation_error_no_errors(self) -> None:
        """Test ConfigurationValidationError without error list."""
        exc = ConfigurationValidationError("Validation failed")
        assert exc.errors == []


# ---------------------------------------------------------------------------
# Settings Object Tests
# ---------------------------------------------------------------------------


class TestSettingsObject:
    """Tests for the Settings object itself."""

    def test_settings_has_all_sections(self) -> None:
        """Test that Settings has all expected sections."""
        settings = Settings(_yaml_file=None)

        assert hasattr(settings, "paperless")
        assert hasattr(settings, "tags")
        assert hasattr(settings, "pipeline")
        assert hasattr(settings, "integration")
        assert hasattr(settings, "auto_processing")
        assert hasattr(settings, "jobs")
        assert hasattr(settings, "gpu")
        assert hasattr(settings, "web")
        assert hasattr(settings, "observability")

    def test_settings_nested_access(self) -> None:
        """Test nested attribute access on Settings."""
        settings = Settings(_yaml_file=None)

        # Deep nesting should work
        assert settings.pipeline.stage1.ocrmypdf.language == "eng"
        assert settings.pipeline.stage1.layout_detection.confidence_threshold == 0.5
        assert settings.pipeline.stage2.marker.use_llm is False
        assert settings.observability.logging.level == LogLevel.INFO
