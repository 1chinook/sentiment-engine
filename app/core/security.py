"""Optional X-API-Key authentication dependency for microservice protection."""

import secrets

from fastapi import Header, HTTPException, Request, status

from app.core.config import Settings, get_settings


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate X-API-Key header when API_KEY is configured in Settings."""
    settings: Settings = getattr(request.app.state, "settings", get_settings())
    configured_key = (settings.API_KEY or "").strip()
    if not configured_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
