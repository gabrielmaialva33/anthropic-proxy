import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from starlette.responses import JSONResponse

from src.api.routes import router
from src.app.utils.error_handler import format_exception
from src.core.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Anthropic-OpenAI Proxy",
    description="""
    A multi-provider proxy service that routes Anthropic Claude API requests to various
    LLM providers (OpenAI, Anthropic, NVIDIA NIM) using LiteLLM.

    ## Key Features

    * Multi-provider support (OpenAI, Anthropic, NVIDIA NIM)
    * Convert Anthropic Claude API format to provider-specific formats
    * Full streaming support with Server-Sent Events
    * Tool/function calling support (always included, provider handles compatibility)
    * Modular architecture with clean separation of concerns
    * Comprehensive error handling and logging

    ## Usage

    Configure your preferred provider via environment variables and use the
    Anthropic Messages API format. The proxy handles all conversions automatically.
    """,
    version="2.0.0",
    docs_url=None,  # Disable the default docs
    redoc_url=None,  # Disable the default redoc
)


# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_details = {
        "error": {
            "type": "http_exception",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    }
    logger.error(f"HTTP Exception: {error_details}")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_details,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_details = format_exception(exc)
    logger.error(f"Unhandled Exception: {error_details}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "unhandled_exception",
                "message": "An unexpected error occurred.",
                "details": error_details
            }
        },
    )


# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path

    response = await call_next(request)

    return response


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
        swagger_ui_parameters={"docExpansion": "none"}  # Make all endpoints collapsed by default
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
