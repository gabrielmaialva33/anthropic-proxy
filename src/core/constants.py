"""
Constants for Anthropic-OpenAI proxy.

This module defines all constants used throughout the application to avoid
magic strings and improve maintainability.
"""


class Constants:
    """Application constants."""

    # Roles
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    # Content types
    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"

    # Tool types
    TOOL_FUNCTION = "function"

    # Stop reasons
    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_STOP_SEQUENCE = "stop_sequence"

    # SSE Event types
    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"
    EVENT_ERROR = "error"

    # Delta types
    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"

    # Message types
    MESSAGE_TYPE = "message"

    # Tool choice types
    TOOL_CHOICE_AUTO = "auto"
    TOOL_CHOICE_ANY = "any"
    TOOL_CHOICE_TOOL = "tool"
