"""CLI module for paperless-ngx-smart-ocr."""

from __future__ import annotations

import typer

from paperless_ngx_smart_ocr import __version__
from paperless_ngx_smart_ocr.observability import LogLevel, configure_logging


app = typer.Typer(
    name="smart-ocr",
    help="Intelligent OCR and Markdown conversion for paperless-ngx.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print version and exit."""
    if value:
        typer.echo(f"smart-ocr version {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--verbose",
        "-V",
        help="Enable verbose (debug) logging.",
    ),
    quiet: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--quiet",
        "-q",
        help="Only show warnings and errors.",
    ),
) -> None:
    """paperless-ngx-smart-ocr CLI."""
    del version  # Handled by callback

    # Determine log level from flags
    if verbose and quiet:
        typer.echo("Error: --verbose and --quiet are mutually exclusive.", err=True)
        raise typer.Exit(1)

    if verbose:
        level = LogLevel.DEBUG
    elif quiet:
        level = LogLevel.WARNING
    else:
        level = LogLevel.INFO

    configure_logging(level=level)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to."),  # noqa: S104
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to."),
    config_file: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file.",
    ),
) -> None:
    """Start the web server with background workers."""
    import uvicorn

    from paperless_ngx_smart_ocr.config import load_settings
    from paperless_ngx_smart_ocr.web import create_app

    settings = load_settings(config_file)
    application = create_app(settings=settings)

    uvicorn.run(application, host=host, port=port)


@app.command()
def process(
    document_id: int = typer.Argument(..., help="Document ID to process."),
) -> None:
    """Process a single document by ID."""
    typer.echo(f"Processing document {document_id}...")
    typer.echo("Not yet implemented.")


@app.command()
def config() -> None:
    """Validate configuration file."""
    typer.echo("Validating configuration...")
    typer.echo("Not yet implemented.")


@app.command(name="post-consume")
def post_consume() -> None:
    """Run in post-consume script mode."""
    typer.echo("Running in post-consume mode...")
    typer.echo("Not yet implemented.")


__all__ = ["app"]
