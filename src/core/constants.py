# Constants for better maintainability
class Constants:
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"
    CONTENT_THINKING = "thinking"

    TOOL_FUNCTION = "function"

    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_ERROR = "error"

    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"

    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"
    DELTA_THINKING = "thinking_delta"

    # Claude-specific fields to strip when forwarding to non-Claude providers
    CLAUDE_ONLY_FIELDS = {"thinking", "cache_control"}

    # JSON Schema fields unsupported by Google Gemini
    GEMINI_UNSUPPORTED_SCHEMA_FIELDS = {
        "additionalProperties",
        "$schema",
        "$ref",
        "$id",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "nullable",
        "discriminator",
        "readOnly",
        "writeOnly",
        "xml",
        "externalDocs",
        "example",
        "examples",
        "deprecated",
        "pattern",
        "patternProperties",
        "minProperties",
        "maxProperties",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentMediaType",
        "contentEncoding",
    }
