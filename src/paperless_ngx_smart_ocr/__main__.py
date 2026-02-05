"""Entry point for running as a module: python -m paperless_ngx_smart_ocr."""

from __future__ import annotations

from paperless_ngx_smart_ocr.cli import app


if __name__ == "__main__":
    app()
