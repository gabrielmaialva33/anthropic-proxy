"""
Configuration management for Anthropic-OpenAI proxy.

This module handles all environment variables and application configuration.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # API Keys
        self.anthropic_api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
        self.nvidia_nim_api_key: Optional[str] = os.environ.get("NVIDIA_NIM_API_KEY")

        # Model Configuration
        self.big_model: str = os.environ.get("BIG_MODEL", "gpt-4o")
        self.small_model: str = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        # Provider Configuration
        self.preferred_provider: str = os.environ.get(
            "PREFERRED_PROVIDER", "openai"
        ).lower()

        # API Configuration
        self.openai_base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL")

        # Server Configuration
        self.server_host: str = os.environ.get("SERVER_HOST", "0.0.0.0")
        self.server_port: int = int(os.environ.get("SERVER_PORT", "8082"))
        self.log_level: str = os.environ.get("LOG_LEVEL", "error").upper()

        # Performance Configuration
        self.max_tokens_limit: int = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
        self.request_timeout: int = int(os.environ.get("REQUEST_TIMEOUT", "90"))

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration."""
        # Check that at least one API key is present
        if not any(
                [
                    self.anthropic_api_key,
                    self.openai_api_key,
                    self.nvidia_nim_api_key,
                ]
        ):
            logger.error(
                "Missing API keys: At least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, or NVIDIA_NIM_API_KEY must be provided"
            )
            logger.error(
                "Please set at least one of these variables in your .env file or environment."
            )
            raise ValueError("No API keys configured")

        # Validate preferred provider
        valid_providers = ["openai", "anthropic", "nvidia"]
        if self.preferred_provider not in valid_providers:
            logger.warning(
                f"Invalid PREFERRED_PROVIDER: {self.preferred_provider}. Using 'openai' as default."
            )
            self.preferred_provider = "openai"

        # Check provider-specific requirements
        if self.preferred_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when PREFERRED_PROVIDER is 'openai'"
            )
        elif self.preferred_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when PREFERRED_PROVIDER is 'anthropic'"
            )
        elif self.preferred_provider == "nvidia" and not self.nvidia_nim_api_key:
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when PREFERRED_PROVIDER is 'nvidia'"
            )

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get the API key for a specific provider."""
        if provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "nvidia":
            return self.nvidia_nim_api_key
        return None

    def get_custom_headers(self) -> dict:
        """Get custom headers from environment variables."""
        headers = {}
        for key, value in os.environ.items():
            if key.startswith("CUSTOM_HEADER_"):
                header_name = key.replace("CUSTOM_HEADER_", "").replace("_", "-")
                headers[header_name] = value
        return headers

    def print_configuration(self):
        """Print configuration summary (without sensitive data)."""
        print("🚀 Anthropic-OpenAI Proxy Configuration")
        print(f"✅ Preferred Provider: {self.preferred_provider}")
        print(f"   OpenAI API Key: {'✓ Set' if self.openai_api_key else '✗ Not Set'}")
        print(
            f"   Anthropic API Key: {'✓ Set' if self.anthropic_api_key else '✗ Not Set'}"
        )
        print(
            f"   NVIDIA NIM API Key: {'✓ Set' if self.nvidia_nim_api_key else '✗ Not Set'}"
        )
        print(f"   Big Model: {self.big_model}")
        print(f"   Small Model: {self.small_model}")
        if self.openai_base_url:
            print(f"   OpenAI Base URL: {self.openai_base_url}")
        print(f"   Server: {self.server_host}:{self.server_port}")
        print(f"   Log Level: {self.log_level}")
        print(f"   Max Tokens Limit: {self.max_tokens_limit}")
        print(f"   Request Timeout: {self.request_timeout}s")
        print()


# Global configuration instance
config = Config()
