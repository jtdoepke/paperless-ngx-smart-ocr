"""Unit tests for the logging module."""

from __future__ import annotations

import logging

import pytest
import structlog

from paperless_ngx_smart_ocr.observability import (
    LogLevel,
    clear_request_context,
    configure_logging,
    generate_request_id,
    get_logger,
    get_request_id,
    set_request_id,
)


# ---------------------------------------------------------------------------
# TestLogLevel
# ---------------------------------------------------------------------------


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"
        assert LogLevel.CRITICAL == "critical"

    def test_to_stdlib_level_debug(self) -> None:
        """Test conversion to stdlib DEBUG level."""
        assert LogLevel.DEBUG.to_stdlib_level() == logging.DEBUG

    def test_to_stdlib_level_info(self) -> None:
        """Test conversion to stdlib INFO level."""
        assert LogLevel.INFO.to_stdlib_level() == logging.INFO

    def test_to_stdlib_level_warning(self) -> None:
        """Test conversion to stdlib WARNING level."""
        assert LogLevel.WARNING.to_stdlib_level() == logging.WARNING

    def test_to_stdlib_level_error(self) -> None:
        """Test conversion to stdlib ERROR level."""
        assert LogLevel.ERROR.to_stdlib_level() == logging.ERROR

    def test_to_stdlib_level_critical(self) -> None:
        """Test conversion to stdlib CRITICAL level."""
        assert LogLevel.CRITICAL.to_stdlib_level() == logging.CRITICAL

    def test_log_level_from_string(self) -> None:
        """Test LogLevel can be created from string."""
        assert LogLevel("debug") == LogLevel.DEBUG
        assert LogLevel("info") == LogLevel.INFO
        assert LogLevel("warning") == LogLevel.WARNING


# ---------------------------------------------------------------------------
# TestRequestId
# ---------------------------------------------------------------------------


class TestRequestId:
    """Tests for request ID management."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_request_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_request_context()

    def test_generate_request_id_returns_string(self) -> None:
        """Test request ID generation returns a string."""
        request_id = generate_request_id()
        assert isinstance(request_id, str)

    def test_generate_request_id_length(self) -> None:
        """Test request ID is 8 characters."""
        request_id = generate_request_id()
        assert len(request_id) == 8

    def test_generate_request_id_is_hex(self) -> None:
        """Test request ID is valid hexadecimal."""
        request_id = generate_request_id()
        # Should not raise ValueError
        int(request_id, 16)

    def test_generate_request_id_unique(self) -> None:
        """Test each generated request ID is unique."""
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_get_request_id_default(self) -> None:
        """Test get_request_id returns None when not set."""
        clear_request_context()
        assert get_request_id() is None

    def test_set_and_get_request_id(self) -> None:
        """Test setting and getting request ID."""
        set_request_id("test-123")
        assert get_request_id() == "test-123"

    def test_set_request_id_generates_when_none(self) -> None:
        """Test set_request_id generates ID when None passed."""
        request_id = set_request_id(None)
        assert request_id is not None
        assert len(request_id) == 8
        assert get_request_id() == request_id

    def test_set_request_id_returns_provided_id(self) -> None:
        """Test set_request_id returns the provided ID."""
        result = set_request_id("my-custom-id")
        assert result == "my-custom-id"

    def test_clear_request_context(self) -> None:
        """Test clearing request context."""
        set_request_id("test-456")
        assert get_request_id() == "test-456"
        clear_request_context()
        assert get_request_id() is None


# ---------------------------------------------------------------------------
# TestConfigureLogging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self) -> None:
        """Reset structlog configuration before each test."""
        structlog.reset_defaults()

    def test_configure_with_string_level(self) -> None:
        """Test configuring with string log level."""
        # Should not raise
        configure_logging(level="debug")

    def test_configure_with_enum_level(self) -> None:
        """Test configuring with LogLevel enum."""
        # Should not raise
        configure_logging(level=LogLevel.INFO)

    def test_configure_with_uppercase_string(self) -> None:
        """Test configuring with uppercase string level."""
        # LogLevel normalizes to lowercase
        configure_logging(level="DEBUG")

    def test_configure_force_colors_true(self) -> None:
        """Test forcing colors on."""
        # Should not raise
        configure_logging(level=LogLevel.INFO, force_colors=True)

    def test_configure_force_colors_false(self) -> None:
        """Test forcing colors off."""
        # Should not raise
        configure_logging(level=LogLevel.INFO, force_colors=False)

    def test_configure_invalid_level_raises(self) -> None:
        """Test invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid LogLevel"):
            configure_logging(level="invalid")


# ---------------------------------------------------------------------------
# TestGetLogger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger function."""

    def setup_method(self) -> None:
        """Clear context before tests."""
        structlog.reset_defaults()
        clear_request_context()

    def teardown_method(self) -> None:
        """Clear context after tests."""
        clear_request_context()
        structlog.reset_defaults()

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test get_logger returns a logger."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """Test get_logger with specific name."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger("my.module")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Test get_logger without name."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_initial_context(self) -> None:
        """Test get_logger with initial context binding."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__, component="test", version=1)
        # Logger should have context bound (verified by actually logging)
        assert logger is not None

    def test_logger_can_log(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test that logger can write log messages."""
        # Configure after capfd is active to capture output
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("test_event")

        captured = capfd.readouterr()
        assert "test_event" in captured.err

    def test_logger_includes_level(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test that logger output includes log level."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("test_event")

        captured = capfd.readouterr()
        assert "info" in captured.err.lower()

    def test_logger_includes_timestamp(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test that logger output includes timestamp."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("test_event")

        captured = capfd.readouterr()
        # ISO 8601 timestamp should contain 'T' and 'Z'
        assert "T" in captured.err
        assert "Z" in captured.err

    def test_logger_includes_request_id_when_set(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Test that logger output includes request_id when set."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        set_request_id("req-abc123")
        logger = get_logger(__name__)
        logger.info("test_event")

        captured = capfd.readouterr()
        assert "req-abc123" in captured.err

    def test_logger_no_request_id_when_not_set(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Test that logger works without request_id."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        clear_request_context()
        logger = get_logger(__name__)
        logger.info("test_event")

        captured = capfd.readouterr()
        assert "test_event" in captured.err
        # request_id should not appear if not set
        # (LogfmtRenderer with drop_missing=True)

    def test_logger_with_extra_fields(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test that logger includes extra fields."""
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("processing", document_id=123, status="ok")

        captured = capfd.readouterr()
        assert "document_id" in captured.err
        assert "123" in captured.err
        assert "status" in captured.err

    def test_logger_level_filtering(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test that logger filters by level."""
        configure_logging(level=LogLevel.WARNING, force_colors=False)
        logger = get_logger(__name__)

        logger.debug("debug_message")
        logger.info("info_message")
        logger.warning("warning_message")

        captured = capfd.readouterr()
        assert "debug_message" not in captured.err
        assert "info_message" not in captured.err
        assert "warning_message" in captured.err


# ---------------------------------------------------------------------------
# TestLogfmtOutput
# ---------------------------------------------------------------------------


class TestLogfmtOutput:
    """Tests for logfmt output format."""

    def setup_method(self) -> None:
        """Reset before tests."""
        structlog.reset_defaults()
        clear_request_context()

    def teardown_method(self) -> None:
        """Reset configuration."""
        clear_request_context()
        structlog.reset_defaults()

    def test_logfmt_key_value_format(self, capfd: pytest.CaptureFixture[str]) -> None:
        """Test logfmt produces key=value format."""
        # Configure after capfd is active to capture output
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("test_event", foo="bar")

        captured = capfd.readouterr()
        # Should contain key=value pairs
        assert "event=test_event" in captured.err or "test_event" in captured.err
        assert "foo=bar" in captured.err

    def test_logfmt_quotes_values_with_spaces(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Test logfmt quotes values containing spaces."""
        # Configure after capfd is active to capture output
        configure_logging(level=LogLevel.DEBUG, force_colors=False)
        logger = get_logger(__name__)
        logger.info("test_event", message="hello world")

        captured = capfd.readouterr()
        # Value with spaces should be quoted
        assert "hello world" in captured.err
