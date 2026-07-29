"""Shared bearer-token authentication for the Hold'em API and its clients."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPOSITORY_ROOT / ".env"
API_TOKEN_ENVIRONMENT_VARIABLE = "HOLDEM_API_TOKEN"

load_dotenv(ENV_PATH)


def api_token() -> str:
    token = os.environ.get(API_TOKEN_ENVIRONMENT_VARIABLE, "").strip()
    if not token:
        raise RuntimeError(
            f"{API_TOKEN_ENVIRONMENT_VARIABLE} must be set in {ENV_PATH} "
            "or in the process environment."
        )
    return token


CONFIGURED_API_TOKEN = api_token()


def api_authorization_headers() -> dict[str, str]:
    """Return the authorization header used by bundled local API clients."""
    return {"Authorization": f"Bearer {CONFIGURED_API_TOKEN}"}


def require_api_token(request: Request) -> None:
    """Reject requests that do not present the configured bearer token."""
    scheme, credentials = get_authorization_scheme_param(
        request.headers.get("Authorization")
    )
    authenticated = (
        scheme.lower() == "bearer"
        and bool(credentials)
        and secrets.compare_digest(credentials, CONFIGURED_API_TOKEN)
    )
    if not authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
