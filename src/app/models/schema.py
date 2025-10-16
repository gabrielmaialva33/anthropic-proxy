"""
Data models for the API
"""
import os
from typing import List, Dict, Any, Optional, Union, Literal

from pydantic import BaseModel, field_validator

PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Default model configurations
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4o")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4o-mini")


def _validate_model(v, info):
    original_model = v
    preferred_provider = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

    if preferred_provider == "nvidia":
        if 'haiku' in v.lower():
            new_model = f"nvidia_nim/{SMALL_MODEL}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
        elif 'sonnet' in v.lower():
            new_model = f"nvidia_nim/{BIG_MODEL}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
        elif not v.startswith('nvidia_nim/'):
            new_model = f"nvidia_nim/{v}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
    elif preferred_provider == "openai":
        if v.startswith('anthropic/'):
            v = v[10:]  # Remove 'anthropic/' prefix

        if 'haiku' in v.lower():
            new_model = f"openai/{SMALL_MODEL}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
        elif 'sonnet' in v.lower():
            new_model = f"openai/{BIG_MODEL}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
        elif not v.startswith('openai/'):
            new_model = f"openai/{v}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model
    else:  # anthropic
        if not v.startswith('anthropic/'):
            new_model = f"anthropic/{v}"
            print(f" MODEL MAPPING: {original_model} → {new_model}")
            v = new_model

    values = info.data
    if isinstance(values, dict):
        values['original_model'] = original_model
    return v


# Content block types
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: Optional[bool] = False


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model(cls, v, info):
        return _validate_model(v, info)


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model(cls, v, info):
        return _validate_model(v, info)


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


class LLMProvider(BaseModel):
    """Configuration for an LLM provider"""
    name: str
    api_key_env: str
    base_url: Optional[str] = None
    default_model_mappings: Dict[str, str] = {}
    retry_config: Optional[Dict[str, Any]] = None


class ProxyConfig(BaseModel):
    """Global configuration for the proxy"""
    providers: List[LLMProvider]
    default_provider: str
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    max_retries: int = 3
    timeout_seconds: int = 30
    debug_mode: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    extra_body: Optional[Dict[str, Any]] = None
