"""CLI module for paperless-ngx-smart-ocr."""

from __future__ import annotations

import typer

from paperless_ngx_smart_ocr import __version__


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
) -> None:
    """paperless-ngx-smart-ocr CLI."""


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to."),  # noqa: S104
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to."),
) -> None:
    """Start the web server with background workers."""
    typer.echo(f"Starting server on {host}:{port}...")
    typer.echo("Not yet implemented.")


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
