from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.routes import router
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Claude-OpenAI Proxy",
    description="""
    A proxy service that enables seamless integration between Anthropic Claude and OpenAI APIs.
    This service allows applications built for one API to be used with the other.

    ## Key Features

    * Convert Anthropic Claude API requests to OpenAI format and vice versa
    * Support for streaming responses
    * Tool/function calling compatibility
    * Proper error handling and conversion
    * Transparent model mapping

    ## Usage

    Replace your API endpoint with this proxy's endpoint and continue using your 
    client application as before.
    """,
    version="1.1.0",
    docs_url=None,  # Disable the default docs
    redoc_url=None,  # Disable the default redoc
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
