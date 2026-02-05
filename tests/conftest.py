"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDFS_DIR = FIXTURES_DIR / "pdfs"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def pdfs_dir() -> Path:
    """Return the path to the PDF fixtures directory."""
    return PDFS_DIR
