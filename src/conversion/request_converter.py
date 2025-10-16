"""
Request converter for transforming Anthropic API requests to LiteLLM format.

This module handles the conversion of Anthropic API requests to a format
compatible with LiteLLM, which can then route to various providers.

IMPORTANT: This converter NEVER filters tools based on model capabilities.
Tools are always included in the request if provided. The provider is
responsible for handling tools appropriately.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union

from src.core.constants import Constants
from src.core.config import config
from src.app.models.schema import MessagesRequest

logger = logging.getLogger(__name__)


def convert_system_prompt(system_prompt) -> Optional[Dict[str, str]]:
    """
    Convert Anthropic system prompt to LiteLLM format.

    Args:
        system_prompt: System prompt (string or list of blocks)

    Returns:
        System message dict or None
    """
    if not system_prompt:
        return None

    if isinstance(system_prompt, str):
        return {
            "role": Constants.ROLE_SYSTEM,
            "content": system_prompt
        }
    elif isinstance(system_prompt, list):
        # Concatenate text blocks from system prompt
        system_text = ""
        for block in system_prompt:
            if hasattr(block, 'type') and block.type == Constants.CONTENT_TEXT:
                system_text += block.text + "\n\n"
            elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                system_text += block.get("text", "") + "\n\n"

        if system_text:
            return {
                "role": Constants.ROLE_SYSTEM,
                "content": system_text.strip()
            }

    return None


def parse_tool_result_content(content) -> str:
    """
    Parse tool result content into a string representation.

    Args:
        content: Tool result content (can be string, list, dict, or None)

    Returns:
        String representation of the content
    """
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"


def convert_messages(messages) -> List[Dict[str, Any]]:
    """
    Convert Anthropic messages to LiteLLM format.

    Args:
        messages: List of Anthropic message objects

    Returns:
        List of LiteLLM-compatible message dicts
    """
    litellm_messages = []

    for idx, msg in enumerate(messages):
        content = msg.content

        # Simple string content
        if isinstance(content, str):
            litellm_messages.append({
                "role": msg.role,
                "content": content
            })
            continue

        # Complex content (list of blocks)
        # Check if message contains tool results
        has_tool_result = any(
            hasattr(block, "type") and block.type == Constants.CONTENT_TOOL_RESULT
            for block in content
        )

        if msg.role == Constants.ROLE_USER and has_tool_result:
            # Flatten tool results to text for user messages
            text_content = ""
            for block in content:
                if hasattr(block, "type"):
                    if block.type == Constants.CONTENT_TEXT:
                        text_content += block.text + "\n"
                    elif block.type == Constants.CONTENT_TOOL_RESULT:
                        tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                        result_content = parse_tool_result_content(
                            block.content if hasattr(block, "content") else None
                        )
                        text_content += f"Tool result for {tool_id}:\n{result_content}\n"

            litellm_messages.append({
                "role": Constants.ROLE_USER,
                "content": text_content.strip()
            })
        else:
            # Process complex content blocks
            processed_content = []
            for block in content:
                if hasattr(block, "type"):
                    if block.type == Constants.CONTENT_TEXT:
                        processed_content.append({
                            "type": Constants.CONTENT_TEXT,
                            "text": block.text
                        })
                    elif block.type == Constants.CONTENT_IMAGE:
                        processed_content.append({
                            "type": Constants.CONTENT_IMAGE,
                            "source": block.source
                        })
                    elif block.type == Constants.CONTENT_TOOL_USE:
                        processed_content.append({
                            "type": Constants.CONTENT_TOOL_USE,
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block.type == Constants.CONTENT_TOOL_RESULT:
                        processed_content_block = {
                            "type": Constants.CONTENT_TOOL_RESULT,
                            "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                        }
                        processed_content_block["content"] = [{
                            "type": Constants.CONTENT_TEXT,
                            "text": parse_tool_result_content(
                                block.content if hasattr(block, "content") else None
                            )
                        }]
                        processed_content.append(processed_content_block)

            litellm_messages.append({
                "role": msg.role,
                "content": processed_content
            })

    return litellm_messages


def convert_tools(tools) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tools to OpenAI function format.

    IMPORTANT: This function ALWAYS returns the tools if provided.
    It does NOT filter based on model capabilities.

    Args:
        tools: List of Anthropic tool objects

    Returns:
        List of OpenAI-formatted tool dicts or None
    """
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        # Convert tool object to dict if needed
        if hasattr(tool, 'dict'):
            tool_dict = tool.dict()
        else:
            tool_dict = tool

        # Build OpenAI function format
        openai_tool = {
            "type": Constants.TOOL_FUNCTION,
            Constants.TOOL_FUNCTION: {
                "name": tool_dict["name"],
                "description": tool_dict.get("description", ""),
                "parameters": tool_dict["input_schema"]
            }
        }
        openai_tools.append(openai_tool)

    logger.debug(f"Converted {len(openai_tools)} tools to OpenAI format")
    return openai_tools


def convert_tool_choice(tool_choice) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Args:
        tool_choice: Anthropic tool choice object

    Returns:
        OpenAI-formatted tool choice or None
    """
    if not tool_choice:
        return None

    # Convert to dict if needed
    if hasattr(tool_choice, 'dict'):
        tool_choice_dict = tool_choice.dict()
    else:
        tool_choice_dict = tool_choice

    choice_type = tool_choice_dict.get("type")

    if choice_type == Constants.TOOL_CHOICE_AUTO:
        return Constants.TOOL_CHOICE_AUTO
    elif choice_type == Constants.TOOL_CHOICE_ANY:
        return Constants.TOOL_CHOICE_ANY
    elif choice_type == Constants.TOOL_CHOICE_TOOL and "name" in tool_choice_dict:
        # Specific tool choice
        return {
            "type": Constants.TOOL_FUNCTION,
            Constants.TOOL_FUNCTION: {
                "name": tool_choice_dict["name"]
            }
        }
    else:
        # Default to auto
        return Constants.TOOL_CHOICE_AUTO


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """
    Convert an Anthropic API request to LiteLLM format.

    IMPORTANT: This function ALWAYS includes tools in the request if provided.
    It does NOT filter tools based on model capabilities. The provider is
    responsible for handling tools appropriately.

    Args:
        anthropic_request: Anthropic MessagesRequest object

    Returns:
        Dict containing LiteLLM-compatible request
    """
    logger.debug(f"Converting Anthropic request for model: {anthropic_request.model}")

    # Start with messages
    messages = []

    # Add system prompt as first message if present
    system_prompt = convert_system_prompt(anthropic_request.system)
    if system_prompt:
        messages.append(system_prompt)

    # Add converted messages
    messages.extend(convert_messages(anthropic_request.messages))

    # Handle max_tokens - cap for OpenAI models
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/"):
        max_tokens = min(max_tokens, config.max_tokens_limit)
        logger.debug(
            f"Capping max_tokens to {config.max_tokens_limit} for OpenAI model "
            f"(original: {anthropic_request.max_tokens})"
        )

    # Build base LiteLLM request
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p is not None:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        litellm_request["top_k"] = anthropic_request.top_k

    # CRITICAL: Always include tools if provided
    # DO NOT filter based on model capabilities
    converted_tools = convert_tools(anthropic_request.tools)
    if converted_tools:
        litellm_request["tools"] = converted_tools
        logger.debug(f"Added {len(converted_tools)} tools to request")

    # Add tool_choice if provided
    converted_tool_choice = convert_tool_choice(anthropic_request.tool_choice)
    if converted_tool_choice:
        litellm_request["tool_choice"] = converted_tool_choice
        logger.debug(f"Added tool_choice to request: {converted_tool_choice}")

    logger.debug(f"Conversion complete. Model: {litellm_request['model']}, "
                f"Messages: {len(litellm_request['messages'])}, "
                f"Tools: {len(converted_tools) if converted_tools else 0}")

    return litellm_request
