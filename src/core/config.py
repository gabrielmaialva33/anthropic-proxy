import os
import sys


# Configuration
class Config:
    # Models that use max_completion_tokens instead of max_tokens
    REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print(
                "Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled."
            )

        self.openai_base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Anthropic passthrough settings
        self.anthropic_base_url = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )
        self.enable_passthrough = (
            os.environ.get("ENABLE_PASSTHROUGH", "true").lower() == "true"
        )

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "120"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Model settings - BIG, MIDDLE, and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

    def is_reasoning_model(self, model: str) -> bool:
        """Check if model uses reasoning/thinking (o1, o3, o4 series)."""
        model_lower = model.lower()
        return any(model_lower.startswith(p) for p in self.REASONING_MODEL_PREFIXES)

    def is_gemini_provider(self) -> bool:
        """Check if the configured provider is Google Gemini."""
        return any(
            x in self.openai_base_url.lower()
            for x in ["googleapis", "generativelanguage", "gemini"]
        )

    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Accept any non-empty key (supports OpenAI, Azure, custom proxies)
        return len(self.openai_api_key.strip()) > 0

    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True

        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self):
        """Get custom headers from environment variables"""
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables
        for env_key, env_value in env_vars.items():
            if env_key.startswith("CUSTOM_HEADER_"):
                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace("_", "-")
                    custom_headers[header_name] = env_value

        return custom_headers


try:
    config = Config()
    print(
        f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'"
    )
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
