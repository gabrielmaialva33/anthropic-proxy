import json
import logging
import uuid

from fastapi import HTTPException, Request

from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest

logger = logging.getLogger(__name__)


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add reasoning/thinking content if present (o1/o3/o4 models)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content_blocks.append(
            {
                "type": Constants.CONTENT_THINKING,
                "thinking": reasoning if isinstance(reasoning, str) else str(reasoning),
            }
        )

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    thinking_block_index = -1  # -1 means no thinking block yet
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    has_thinking = False

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning/thinking delta (o1/o3/o4 models)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if not has_thinking:
                            has_thinking = True
                            thinking_block_index = 0
                            text_block_index = 1
                            # Re-emit block 0 as thinking instead of text
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': thinking_block_index, 'content_block': {'type': Constants.CONTENT_THINKING, 'thinking': ''}}, ensure_ascii=False)}\n\n"
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': reasoning}}, ensure_ascii=False)}\n\n"

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        if has_thinking and text_block_index == 1:
                            # Close thinking block and start text block
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': thinking_block_index}, ensure_ascii=False)}\n\n"
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"
                            has_thinking = False  # Already transitioned
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False,
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have complete initial data
                            if (
                                tool_call["id"]
                                and tool_call["name"]
                                and not tool_call["started"]
                            ):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"

                            # Handle function arguments — send incremental deltas
                            if (
                                "arguments" in function_data
                                and tool_call["started"]
                                and function_data["arguments"] is not None
                            ):
                                args_chunk = function_data["arguments"]
                                tool_call["args_buffer"] += args_chunk

                                # Send each chunk as an incremental delta
                                if args_chunk:
                                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': args_chunk}}, ensure_ascii=False)}\n\n"

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break

    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Close text block
    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    # Close all tool blocks (even incomplete ones to prevent client hangs)
    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    usage_data = {"input_tokens": 0, "output_tokens": 0}
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    thinking_block_index = -1
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}
    has_thinking = False

    try:
        async for line in openai_stream:
            # Check if client disconnected
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        usage = chunk.get("usage", None)
                        if usage:
                            cache_read_input_tokens = 0
                            prompt_tokens_details = (
                                usage.get("prompt_tokens_details") or {}
                            )
                            cache_read_input_tokens = prompt_tokens_details.get(
                                "cached_tokens", 0
                            )
                            usage_data = {
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "cache_read_input_tokens": cache_read_input_tokens,
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning/thinking delta (o1/o3/o4 models)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if not has_thinking:
                            has_thinking = True
                            thinking_block_index = 0
                            text_block_index = 1
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': thinking_block_index, 'content_block': {'type': Constants.CONTENT_THINKING, 'thinking': ''}}, ensure_ascii=False)}\n\n"
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': reasoning}}, ensure_ascii=False)}\n\n"

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        if has_thinking and text_block_index == 1:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': thinking_block_index}, ensure_ascii=False)}\n\n"
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"
                            has_thinking = False
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False,
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have complete initial data
                            if (
                                tool_call["id"]
                                and tool_call["name"]
                                and not tool_call["started"]
                            ):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"

                            # Handle function arguments — send incremental deltas
                            if (
                                "arguments" in function_data
                                and tool_call["started"]
                                and function_data["arguments"] is not None
                            ):
                                args_chunk = function_data["arguments"]
                                tool_call["args_buffer"] += args_chunk

                                # Send each chunk as an incremental delta
                                if args_chunk:
                                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': args_chunk}}, ensure_ascii=False)}\n\n"

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

    except HTTPException as e:
        # Handle cancellation
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return
        else:
            raise
    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Close text block
    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    # Close all tool blocks (even incomplete ones to prevent client hangs)
    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"
