import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base application error mapped to HTTP responses."""

    def __init__(self, message: str, *, error: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.error = error
        self.status_code = status_code
        self.details = details


class NotFoundError(APIError):
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message, error="not_found", status_code=404, details=details)


class BadRequestError(APIError):
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message, error="bad_request", status_code=400, details=details)


def _error_payload(error: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(error=error, message=message, details=details).model_dump()


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.error, exc.message, exc.details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            message = detail.get("message", str(detail))
            error = detail.get("error", "http_error")
            details = detail.get("details")
        else:
            message = str(detail)
            error = "http_error"
            details = None
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(error, message, details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "validation_error",
                "Request validation failed.",
                {"issues": exc.errors()},
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception):
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", "An unexpected server error occurred."),
        )
