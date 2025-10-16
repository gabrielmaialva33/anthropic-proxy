"""
Response converter for transforming OpenAI responses to Anthropic API format.

This module handles the conversion of OpenAI API responses back to Anthropic's
expected response format, including both streaming and non-streaming responses.
"""
import json
import logging
import uuid
from typing import Dict, Any, List, Union, AsyncGenerator

from src.app.models.schema import MessagesRequest, MessagesResponse, Usage
from src.core.constants import Constants

logger = logging.getLogger(__name__)


def convert_stop_reason(finish_reason: str) -> str:
    """
    Convert OpenAI finish_reason to Anthropic stop_reason.

    Args:
        finish_reason: OpenAI finish reason

    Returns:
        Anthropic stop reason
    """
    if finish_reason == "stop":
        return Constants.STOP_END_TURN
    elif finish_reason == "length":
        return Constants.STOP_MAX_TOKENS
    elif finish_reason == "tool_calls":
        return Constants.STOP_TOOL_USE
    else:
        return Constants.STOP_END_TURN


def extract_response_data(openai_response: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Extract response data from OpenAI response object.

    Handles both dict and object responses.

    Args:
        openai_response: OpenAI response (dict or object)

    Returns:
        Dict containing extracted response data
    """
    # Try to extract from object attributes first
    if hasattr(openai_response, 'choices') and hasattr(openai_response, 'usage'):
        choices = openai_response.choices
        message = choices[0].message if choices and len(choices) > 0 else None
        content_text = message.content if message and hasattr(message, 'content') else ""
        tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
        finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
        usage_info = openai_response.usage
        response_id = getattr(openai_response, 'id', f"msg_{uuid.uuid4()}")

        return {
            "content_text": content_text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": usage_info,
            "id": response_id
        }

    # Handle dictionary response
    try:
        response_dict = openai_response if isinstance(openai_response, dict) else openai_response.dict()
    except AttributeError:
        try:
            response_dict = (
                openai_response.model_dump()
                if hasattr(openai_response, 'model_dump')
                else openai_response.__dict__
            )
        except AttributeError:
            response_dict = {
                "id": getattr(openai_response, 'id', f"msg_{uuid.uuid4()}"),
                "choices": getattr(openai_response, 'choices', [{}]),
                "usage": getattr(openai_response, 'usage', {})
            }

    choices = response_dict.get("choices", [{}])
    message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
    content_text = message.get("content", "")
    tool_calls = message.get("tool_calls", None)
    finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
    usage_info = response_dict.get("usage", {})
    response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

    return {
        "content_text": content_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage_info,
        "id": response_id
    }


def convert_tool_calls_to_content(tool_calls, is_claude_model: bool) -> List[Dict[str, Any]]:
    """
    Convert OpenAI tool calls to Anthropic content blocks.

    Args:
        tool_calls: List of tool calls from OpenAI response
        is_claude_model: Whether the target model is a Claude model

    Returns:
        List of content blocks
    """
    content = []

    if not tool_calls:
        return content

    # Normalize to list
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]

    if is_claude_model:
        # Convert to Anthropic tool_use blocks
        for idx, tool_call in enumerate(tool_calls):
            logger.debug(f"Processing tool call {idx}: {tool_call}")

            # Extract tool call information
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")
            else:
                function = getattr(tool_call, "function", None)
                tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = getattr(function, "name", "") if function else ""
                arguments = getattr(function, "arguments", "{}") if function else "{}"

            # Parse arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                    arguments = {"raw": arguments}

            # Add tool use block
            logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
            content.append({
                "type": Constants.CONTENT_TOOL_USE,
                "id": tool_id,
                "name": name,
                "input": arguments
            })
    else:
        # Convert to text for non-Claude models
        logger.debug(f"Converting tool calls to text for non-Claude model")
        tool_text = "\n\nTool usage:\n"

        for idx, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")
            else:
                function = getattr(tool_call, "function", None)
                tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = getattr(function, "name", "") if function else ""
                arguments = getattr(function, "arguments", "{}") if function else "{}"

            if isinstance(arguments, str):
                try:
                    args_dict = json.loads(arguments)
                    arguments_str = json.dumps(args_dict, indent=2)
                except json.JSONDecodeError:
                    arguments_str = arguments
            else:
                arguments_str = json.dumps(arguments, indent=2)

            tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"

        content.append({
            "type": Constants.CONTENT_TEXT,
            "text": tool_text
        })

    return content


def convert_openai_to_anthropic(
        openai_response: Union[Dict[str, Any], Any],
        original_request: MessagesRequest
) -> MessagesResponse:
    """
    Convert an OpenAI response to Anthropic API format.

    Args:
        openai_response: Response from OpenAI API
        original_request: Original Anthropic request

    Returns:
        Anthropic MessagesResponse object
    """
    try:
        # Extract clean model name
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]

        # Check if model is a Claude model
        is_claude_model = clean_model.startswith("claude-")

        # Extract response data
        response_data = extract_response_data(openai_response)

        # Build content blocks
        content = []

        # Add text content block if present
        if response_data["content_text"] is not None and response_data["content_text"] != "":
            content.append({
                "type": Constants.CONTENT_TEXT,
                "text": response_data["content_text"]
            })

        # Add tool calls as content blocks
        tool_content = convert_tool_calls_to_content(
            response_data["tool_calls"],
            is_claude_model
        )

        # Handle appending tool content
        if tool_content:
            if is_claude_model:
                # Add as separate blocks for Claude
                content.extend(tool_content)
            else:
                # Append to text content for non-Claude
                if content and content[0]["type"] == Constants.CONTENT_TEXT:
                    content[0]["text"] += tool_content[0]["text"]
                else:
                    content.extend(tool_content)

        # Extract token counts
        usage_info = response_data["usage"]
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Convert stop reason
        stop_reason = convert_stop_reason(response_data["finish_reason"])

        # Ensure we have at least one content block
        if not content:
            content.append({
                "type": Constants.CONTENT_TEXT,
                "text": ""
            })

        # Build the response object
        anthropic_response = MessagesResponse(
            id=response_data["id"],
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )

        logger.debug(f"Converted response: id={anthropic_response.id}, "
                     f"stop_reason={stop_reason}, "
                     f"blocks={len(content)}")

        return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        # Return an error response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=[{
                "type": Constants.CONTENT_TEXT,
                "text": f"Error converting response: {str(e)}. Please check server logs."
            }],
            stop_reason=Constants.STOP_END_TURN,
            usage=Usage(input_tokens=0, output_tokens=0)
        )


async def handle_streaming(
        response_generator: AsyncGenerator,
        original_request: MessagesRequest
) -> AsyncGenerator[str, None]:
    """
    Handle streaming responses from OpenAI and convert to Anthropic SSE format.

    Converts OpenAI streaming chunks to Anthropic's Server-Sent Events format.

    Args:
        response_generator: Async generator from OpenAI API
        original_request: Original Anthropic request

    Yields:
        Server-Sent Event formatted strings
    """
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Send message_start event
        message_data = {
            'type': Constants.EVENT_MESSAGE_START,
            'message': {
                'id': message_id,
                'type': Constants.MESSAGE_TYPE,
                'role': Constants.ROLE_ASSISTANT,
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps(message_data)}\n\n"

        # Start text content block
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"

        # Send ping event
        yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING})}\n\n"

        # Initialize tracking variables
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        # Process streaming chunks
        async for chunk in response_generator:
            try:
                # Extract token counts
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens

                # Process chunk choices
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Extract delta
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        delta = getattr(choice, 'message', {})

                    finish_reason = getattr(choice, 'finish_reason', None)

                    # Process delta text content
                    delta_content = None
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']

                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': 0, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta_content}})}\n\n"

                    # Process tool calls
                    delta_tool_calls = None
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']

                    if delta_tool_calls:
                        # Close text block if starting tools
                        if tool_index is None:
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"
                            elif accumulated_text and not text_sent and not text_block_closed:
                                text_sent = True
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': 0, 'delta': {'type': Constants.DELTA_TEXT, 'text': accumulated_text}})}\n\n"
                                text_block_closed = True
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"

                        # Normalize to list
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        # Process each tool call
                        for tool_call in delta_tool_calls:
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0

                            # Start new tool if index changed
                            if tool_index is None or current_index != tool_index:
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                # Extract tool information
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")

                                # Start tool use block
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': anthropic_tool_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""

                            # Process tool arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''

                            if arguments:
                                try:
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    args_json = arguments

                                tool_content += args_json if isinstance(args_json, str) else ""
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': anthropic_tool_index, 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': args_json}})}\n\n"

                    # Handle completion
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True

                        # Close any open tool blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': i})}\n\n"

                        # Close text block if still open
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': 0, 'delta': {'type': Constants.DELTA_TEXT, 'text': accumulated_text}})}\n\n"
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"

                        # Convert stop reason
                        stop_reason = convert_stop_reason(finish_reason)

                        # Send message completion events
                        usage = {"output_tokens": output_tokens}
                        yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        # Handle case where we reach end without finish_reason
        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': i})}\n\n"

            # Close text block if needed
            if not text_block_closed:
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"

            # Send completion events
            usage = {"output_tokens": output_tokens}
            yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': Constants.STOP_END_TURN, 'stop_sequence': None}, 'usage': usage})}\n\n"
            yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        # Send error events
        yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
        yield "data: [DONE]\n\n"
