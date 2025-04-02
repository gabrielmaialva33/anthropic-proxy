import time
import traceback
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    error_type: str
    error_code: str
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    http_status: int = 500
    timestamp: float = time.time()
    traceback: Optional[str] = None


class ErrorHandler:
    @staticmethod
    def handle_exception(e: Exception, include_traceback: bool = False) -> ErrorDetail:
        """Convert exceptions to standardized error details"""
        error_detail = ErrorDetail(
            message=str(e),
            error_type=e.__class__.__name__,
            error_code="internal_error",
            http_status=500
        )

        # Handle specific error types
        if hasattr(e, 'status_code'):
            error_detail.http_status = getattr(e, 'status_code')

        if hasattr(e, 'llm_provider'):
            error_detail.provider = getattr(e, 'llm_provider')

        if hasattr(e, 'model'):
            error_detail.model = getattr(e, 'model')

        # Add request_id if available
        if hasattr(e, 'request_id'):
            error_detail.request_id = getattr(e, 'request_id')

        # Customize error codes based on error type
        if isinstance(e, TimeoutError):
            error_detail.error_code = "timeout"
        elif "rate limit" in str(e).lower():
            error_detail.error_code = "rate_limited"
        elif "authentication" in str(e).lower() or "api key" in str(e).lower():
            error_detail.error_code = "authentication_error"
        elif "context length" in str(e).lower() or "token limit" in str(e).lower():
            error_detail.error_code = "context_length_exceeded"

        # Add traceback for debugging if requested
        if include_traceback:
            error_detail.traceback = traceback.format_exc()

        return error_detail

    @staticmethod
    def raise_http_exception(error_detail: ErrorDetail):
        """Raise FastAPI HTTPException from error detail"""
        raise HTTPException(
            status_code=error_detail.http_status,
            detail=error_detail.dict(exclude={"traceback"} if error_detail.traceback else {})
        )
