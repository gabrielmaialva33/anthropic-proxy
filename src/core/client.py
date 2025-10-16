"""
OpenAI client wrapper with cancellation support.

This module provides an async OpenAI client wrapper that handles:
- Request cancellation tracking
- Error classification with user-friendly messages
- Streaming with proper SSE format
- Azure OpenAI support
"""
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any

from fastapi import HTTPException
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai._exceptions import (
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async OpenAI client with cancellation support and error handling."""

    def __init__(
            self,
            api_key: str,
            base_url: str,
            timeout: int = 90,
            api_version: Optional[str] = None,
            custom_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            timeout: Request timeout in seconds
            api_version: Azure API version (if using Azure)
            custom_headers: Custom headers to include in requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}

        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "anthropic-proxy/2.0.0"
        }

        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}

        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            logger.info("Initializing Azure OpenAI client")
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                default_headers=all_headers
            )
        else:
            logger.info(f"Initializing OpenAI client with base URL: {base_url}")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                default_headers=all_headers
            )

        # Track active requests for cancellation
        self.active_requests: Dict[str, asyncio.Event] = {}

    async def create_chat_completion(
            self,
            request: Dict[str, Any],
            request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat completion to OpenAI API with cancellation support.

        Args:
            request: OpenAI chat completion request dict
            request_id: Optional request ID for cancellation tracking

        Returns:
            OpenAI response as dict

        Raises:
            HTTPException: On API errors
        """
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Create task that can be cancelled
            completion_task = asyncio.create_task(
                self.client.chat.completions.create(**request)
            )

            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(
                        status_code=499,
                        detail="Request cancelled by client"
                    )

                completion = await completion_task
            else:
                completion = await completion_task

            # Convert to dict format
            return completion.model_dump()

        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=self.classify_openai_error(str(e))
            )
        except RateLimitError as e:
            raise HTTPException(
                status_code=429,
                detail=self.classify_openai_error(str(e))
            )
        except BadRequestError as e:
            raise HTTPException(
                status_code=400,
                detail=self.classify_openai_error(str(e))
            )
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(
                status_code=status_code,
                detail=self.classify_openai_error(str(e))
            )
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    async def create_chat_completion_stream(
            self,
            request: Dict[str, Any],
            request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Send streaming chat completion to OpenAI API with cancellation support.

        Args:
            request: OpenAI chat completion request dict
            request_id: Optional request ID for cancellation tracking

        Yields:
            SSE-formatted chunks

        Raises:
            HTTPException: On API errors
        """
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True

            logger.debug(f"Starting streaming completion for request_id: {request_id}")

            # Create the streaming completion
            streaming_completion = await self.client.chat.completions.create(**request)

            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        logger.info(f"Request {request_id} cancelled by client")
                        raise HTTPException(
                            status_code=499,
                            detail="Request cancelled by client"
                        )

                # Convert chunk to SSE format
                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"

            # Signal end of stream
            yield "data: [DONE]"
            logger.debug(f"Streaming completion finished for request_id: {request_id}")

        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=self.classify_openai_error(str(e))
            )
        except RateLimitError as e:
            raise HTTPException(
                status_code=429,
                detail=self.classify_openai_error(str(e))
            )
        except BadRequestError as e:
            raise HTTPException(
                status_code=400,
                detail=self.classify_openai_error(str(e))
            )
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(
                status_code=status_code,
                detail=self.classify_openai_error(str(e))
            )
        except Exception as e:
            logger.error(f"Unexpected error in streaming completion: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """
        Provide specific error guidance for common OpenAI API issues.

        Args:
            error_detail: Error details from OpenAI

        Returns:
            User-friendly error message
        """
        error_str = str(error_detail).lower()

        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or \
                "country, region, or territory not supported" in error_str:
            return ("OpenAI API is not available in your region. "
                    "Consider using a VPN or Azure OpenAI service.")

        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."

        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return ("Rate limit exceeded. Please wait and try again, "
                    "or upgrade your API plan.")

        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return ("Model not found. Please check your BIG_MODEL and "
                    "SMALL_MODEL configuration.")

        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."

        # Context length exceeded
        if "context_length_exceeded" in error_str or "maximum context length" in error_str:
            return ("Context length exceeded. Please reduce the size of your "
                    "messages or max_tokens parameter.")

        # Default: return original message
        return str(error_detail)

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel an active request by request_id.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if request was cancelled, False if not found
        """
        if request_id in self.active_requests:
            logger.info(f"Cancelling request: {request_id}")
            self.active_requests[request_id].set()
            return True
        logger.warning(f"Request {request_id} not found for cancellation")
        return False
