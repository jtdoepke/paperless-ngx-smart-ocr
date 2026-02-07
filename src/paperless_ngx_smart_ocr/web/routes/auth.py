"""Login and logout routes for cookie-based authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from paperless_ngx_smart_ocr.web.auth import (
    AUTH_COOKIE_MAX_AGE,
    AUTH_COOKIE_NAME,
    validate_token,
)


if TYPE_CHECKING:
    from starlette.responses import Response

    from paperless_ngx_smart_ocr.config import Settings


__all__ = ["router"]

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> Response:
    """Render the login page.

    Redirects to ``/`` if the user already has a valid cookie.

    Args:
        request: The incoming HTTP request.

    Returns:
        Login page HTML or redirect.
    """
    if request.cookies.get(AUTH_COOKIE_NAME):
        return RedirectResponse(url="/", status_code=302)

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        "login.html",
        {"request": request, "error": None},
    )


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    token: str = Form(...),
) -> Response:
    """Validate token and set auth cookie.

    Args:
        request: The incoming HTTP request.
        token: The paperless-ngx API token from the form.

    Returns:
        Redirect to ``/`` on success, re-rendered login page on failure.
    """
    settings: Settings = request.app.state.settings
    base_url = settings.paperless.url

    if await validate_token(base_url, token):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key=AUTH_COOKIE_NAME,
            value=token,
            max_age=AUTH_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=settings.web.cookie_secure,
        )
        return response

    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        "login.html",
        {
            "request": request,
            "error": "Invalid token or cannot reach paperless-ngx.",
        },
    )


@router.post("/logout")
async def logout(request: Request) -> Response:
    """Clear auth cookie and redirect to login.

    Args:
        request: The incoming HTTP request.

    Returns:
        Redirect to ``/login``.
    """
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=request.app.state.settings.web.cookie_secure,
    )
    return response
