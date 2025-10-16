"""
API routes for the application
"""
import json
import litellm
import logging
import os
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import HTMLResponse

from src.app.models.schema import (
    MessagesRequest,
    TokenCountRequest,
    TokenCountResponse,
    ChatCompletionRequest
)
from src.app.services.converter import (
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
    handle_streaming
)
from src.app.utils.helpers import log_request_beautifully

# Create router
router = APIRouter()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Get logger
logger = logging.getLogger(__name__)


@router.post("/v1/messages")
async def create_message(
        request: MessagesRequest,
        raw_request: Request
):
    """
    Main endpoint for handling messages API requests
    """
    # Get the raw request body for debugging
    body = await raw_request.body()
    body_json = json.loads(body.decode('utf-8'))

    # Extract model information
    original_model = body_json.get("model", "unknown")
    display_model = original_model
    if "/" in display_model:
        display_model = display_model.split("/")[-1]

    # Extract the clean model name without provider prefix
    clean_model = request.model
    if clean_model.startswith("anthropic/"):
        clean_model = clean_model[len("anthropic/"):]
    elif clean_model.startswith("openai/"):
        clean_model = clean_model[len("openai/"):]

    # Log request details
    logger.debug(f" PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

    # Convert the request to LiteLLM format
    litellm_request = convert_anthropic_to_litellm(request)

    # Set the appropriate API key based on the model provider
    if request.model.startswith("openai/"):
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required for OpenAI models but was not provided"
            )
        litellm_request["api_key"] = OPENAI_API_KEY
        logger.debug(f"Using OpenAI API key for model: {request.model}")
    else:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="Anthropic API key is required for Anthropic models but was not provided"
            )
        litellm_request["api_key"] = ANTHROPIC_API_KEY
        logger.debug(f"Using Anthropic API key for model: {request.model}")

    # Special handling for OpenAI models with complex content
    if "openai" in litellm_request["model"] and "messages" in litellm_request:
        logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

        # Process each message
        for i, msg in enumerate(litellm_request["messages"]):
            # Special handling for tool results in messages
            if "content" in msg and isinstance(msg["content"], list):
                is_only_tool_result = True
                for block in msg["content"]:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        is_only_tool_result = False
                        break

                # If message contains only tool results, flatten to text
                if is_only_tool_result and len(msg["content"]) > 0:
                    logger.warning(f"Found message with only tool_result content - special handling required")

                    all_text = ""
                    for block in msg["content"]:
                        all_text += "Tool Result:\n"
                        result_content = block.get("content", [])

                        if isinstance(result_content, list):
                            for item in result_content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    all_text += item.get("text", "") + "\n"
                                elif isinstance(item, dict):
                                    try:
                                        item_text = item.get("text", json.dumps(item))
                                        all_text += item_text + "\n"
                                    except:
                                        all_text += str(item) + "\n"
                        elif isinstance(result_content, str):
                            all_text += result_content + "\n"
                        else:
                            try:
                                all_text += json.dumps(result_content) + "\n"
                            except:
                                all_text += str(result_content) + "\n"

                    litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                    logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                    continue  # Skip normal processing for this message

            # Convert complex content to text for OpenAI models
            if "content" in msg:
                if isinstance(msg["content"], list):
                    text_content = ""
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_content += block.get("text", "") + "\n"
                            elif block.get("type") == "tool_result":
                                tool_id = block.get("tool_use_id", "unknown")
                                text_content += f"[Tool Result ID: {tool_id}]\n"

                                # Process the result content
                                result_content = block.get("content", [])
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            text_content += item.get("text", "") + "\n"
                                        elif isinstance(item, dict):
                                            if "text" in item:
                                                text_content += item.get("text", "") + "\n"
                                            else:
                                                try:
                                                    text_content += json.dumps(item) + "\n"
                                                except:
                                                    text_content += str(item) + "\n"
                                elif isinstance(result_content, dict):
                                    if result_content.get("type") == "text":
                                        text_content += result_content.get("text", "") + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                elif isinstance(result_content, str):
                                    text_content += result_content + "\n"
                                else:
                                    try:
                                        text_content += json.dumps(result_content) + "\n"
                                    except:
                                        text_content += str(result_content) + "\n"
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_id = block.get("id", "unknown")
                                tool_input = json.dumps(block.get("input", {}))
                                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                            elif block.get("type") == "image":
                                text_content += "[Image content - not displayed in text format]\n"

                    if not text_content.strip():
                        text_content = "..."

                    litellm_request["messages"][i]["content"] = text_content.strip()
                elif msg["content"] is None:
                    litellm_request["messages"][i]["content"] = "..."  # Empty content not allowed

            # Remove unsupported fields
            for key in list(msg.keys()):
                if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                    logger.warning(f"Removing unsupported field from message: {key}")
                    del msg[key]

        # Final validation of message formats
        for i, msg in enumerate(litellm_request["messages"]):
            logger.debug(
                f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
            if isinstance(msg.get("content"), list):
                logger.warning(
                    f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
            elif msg.get("content") is None:
                logger.warning(f"Message {i} has None content - replacing with placeholder")
                litellm_request["messages"][i]["content"] = "..."  # Fallback placeholder

    # Log model details
    logger.debug(
        f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

    # Count messages and tools for logging
    num_tools = len(request.tools) if request.tools else 0

    # Handle streaming responses
    if request.stream:
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200  # Assuming success at this point
        )

        response_generator = await litellm.acompletion(**litellm_request)

        return StreamingResponse(
            handle_streaming(response_generator, request),
            media_type="text/event-stream"
        )

    # Handle non-streaming responses
    else:
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200  # Assuming success at this point
        )

        start_time = time.time()
        litellm_response = litellm.completion(**litellm_request)
        logger.debug(
            f" RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

        anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

        return anthropic_response


@router.post("/v1/messages/count_tokens")
async def count_tokens(
        request: TokenCountRequest,
        raw_request: Request
):
    """
    Endpoint to count tokens for a request
    """
    # Extract model information
    original_model = request.original_model or request.model
    display_model = original_model
    if "/" in display_model:
        display_model = display_model.split("/")[-1]

    # Extract clean model name
    clean_model = request.model
    if clean_model.startswith("anthropic/"):
        clean_model = clean_model[len("anthropic/"):]
    elif clean_model.startswith("openai/"):
        clean_model = clean_model[len("openai/"):]

    # Convert to LiteLLM request format
    converted_request = convert_anthropic_to_litellm(
        MessagesRequest(
            model=request.model,
            max_tokens=100,  # Arbitrary value not used for token counting
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking
        )
    )

    from litellm import token_counter

    # Count messages and tools for logging
    num_tools = len(request.tools) if request.tools else 0

    # Log the request
    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        converted_request.get('model'),
        len(converted_request['messages']),
        num_tools,
        200  # Assuming success at this point
    )

    # Count tokens
    token_count = token_counter(
        model=converted_request["model"],
        messages=converted_request["messages"],
    )

    return TokenCountResponse(input_tokens=token_count)


@router.get("/")
async def root():
    """
    Root endpoint to verify the server is running
    """
    return {"message": "Anthropic Proxy for LiteLLM"}


async def handle_openai_streaming(response_generator):
    """
    Handle streaming responses from LiteLLM and convert to OpenAI format.
    """
    async for chunk in response_generator:
        yield f"data: {chunk.json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request
):
    """
    Main endpoint for handling OpenAI-compatible chat completions.
    """
    litellm_request = request.dict()
    litellm_request["api_key"] = OPENAI_API_KEY

    if request.stream:
        response_generator = await litellm.acompletion(**litellm_request)
        return StreamingResponse(
            handle_openai_streaming(response_generator),
            media_type="text/event-stream"
        )
    else:
        response = litellm.completion(**litellm_request)
        return response


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard():
    """Render a simple metrics dashboard for monitoring proxy performance"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Claude-OpenAI Proxy Metrics</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .metric-card {
                @apply bg-white rounded-lg shadow p-4 hover:shadow-lg transition-shadow;
            }
            .metric-value {
                @apply text-3xl font-bold text-blue-600;
            }
            .metric-label {
                @apply text-gray-500 text-sm uppercase;
            }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-2xl font-bold text-gray-800 mb-6">Claude-OpenAI Proxy Metrics</h1>

            <div id="metrics-content" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <!-- Metrics will be populated here -->
                <div class="metric-card">
                    <p class="metric-value" id="total-requests">...</p>
                    <p class="metric-label">Total Requests</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="success-rate">...</p>
                    <p class="metric-label">Success Rate</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="avg-latency">...</p>
                    <p class="metric-label">Avg. Latency (ms)</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value" id="cache-hit-rate">...</p>
                    <p class="metric-label">Cache Hit Rate</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-white rounded-lg shadow p-4">
                    <h2 class="text-lg font-semibold mb-4">Requests per Minute</h2>
                    <canvas id="requests-chart"></canvas>
                </div>
                <div class="bg-white rounded-lg shadow p-4">
                    <h2 class="text-lg font-semibold mb-4">Response Time (ms)</h2>
                    <canvas id="latency-chart"></canvas>
                </div>
            </div>

            <div class="mt-8 bg-white rounded-lg shadow p-4">
                <h2 class="text-lg font-semibold mb-4">Recent Requests</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full table-auto">
                        <thead>
                            <tr class="bg-gray-100">
                                <th class="px-4 py-2 text-left">Time</th>
                                <th class="px-4 py-2 text-left">Source Model</th>
                                <th class="px-4 py-2 text-left">Target Model</th>
                                <th class="px-4 py-2 text-left">Status</th>
                                <th class="px-4 py-2 text-left">Duration</th>
                            </tr>
                        </thead>
                        <tbody id="requests-table">
                            <!-- Request rows will be populated here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Fetch metrics data every 5 seconds
            function fetchMetrics() {
                fetch('/api/metrics-data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-requests').textContent = data.total_requests;
                        document.getElementById('success-rate').textContent = data.success_rate + '%';
                        document.getElementById('avg-latency').textContent = data.avg_latency + 'ms';
                        document.getElementById('cache-hit-rate').textContent = data.cache_hit_rate + '%';

                        // Update charts
                        updateCharts(data);

                        // Update request table
                        updateRequestTable(data.recent_requests);
                    })
                    .catch(error => console.error('Error fetching metrics:', error));
            }

            // Initialize and fetch initial data
            fetchMetrics();
            setInterval(fetchMetrics, 5000);

            // Chart initialization code would go here
            function updateCharts(data) {
                // Implementation for updating charts with new data
            }

            function updateRequestTable(requests) {
                const tableBody = document.getElementById('requests-table');
                tableBody.innerHTML = '';

                requests.forEach(req => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="border px-4 py-2">${req.timestamp}</td>
                        <td class="border px-4 py-2">${req.source_model}</td>
                        <td class="border px-4 py-2">${req.target_model}</td>
                        <td class="border px-4 py-2">
                            <span class="px-2 py-1 rounded ${req.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                ${req.status}
                            </span>
                        </td>
                        <td class="border px-4 py-2">${req.duration}ms</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/api/metrics-data")
async def get_metrics_data():
    """Return metrics data for the dashboard"""
    # This would be implemented to return real metrics from your monitoring system
    # For now, returning sample data
    return {
        "total_requests": 1258,
        "success_rate": 99.2,
        "avg_latency": 245,
        "cache_hit_rate": 42.5,
        "requests_per_minute": [12, 15, 10, 8, 14, 18, 22, 25, 20, 18],
        "latency_over_time": [230, 245, 260, 220, 210, 240, 250, 270, 230, 220],
        "recent_requests": [
            {
                "timestamp": "2023-06-14 15:42:33",
                "source_model": "claude-3-sonnet",
                "target_model": "gpt-4o",
                "status": "success",
                "duration": 245
            },
            {
                "timestamp": "2023-06-14 15:41:58",
                "source_model": "claude-3-haiku",
                "target_model": "gpt-4o-mini",
                "status": "success",
                "duration": 134
            },
            {
                "timestamp": "2023-06-14 15:41:22",
                "source_model": "claude-3-sonnet",
                "target_model": "gpt-4o",
                "status": "error",
                "duration": 320
            }
        ]
    }
