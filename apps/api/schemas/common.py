"""Shared API response and error schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    error: str
    message: str
    details: dict | None = None


class LinkRef(BaseModel):
    """Reference to a related API resource."""

    rel: str
    href: str


class TimestampedResponse(BaseModel):
    """Mixin-style base for responses that include a timestamp."""

    timestamp: datetime = Field(default_factory=datetime.now)
