from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Claude on OpenAI Proxy",
    description="A proxy service that converts Anthropic Claude API requests to OpenAI API format",
    version="1.0.0"
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
