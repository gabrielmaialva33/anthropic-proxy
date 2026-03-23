from src.core.config import config


class ModelManager:
    # Known provider prefixes that should be passed through
    PASSTHROUGH_PREFIXES = (
        "gpt-",
        "o1-",
        "o3-",
        "o4-",  # OpenAI
        "ep-",
        "doubao-",
        "deepseek-",  # Doubao/DeepSeek
        "meta/",
        "mistralai/",
        "microsoft/",
        "google/",
        "nvidia/",  # OpenRouter/NVIDIA
        "anthropic/",
        "cohere/",
        "perplexity/",  # OpenRouter providers
        "openrouter/",  # OpenRouter meta models
        "qwen/",
        "databricks/",  # Other providers
    )

    def __init__(self, cfg):
        self.config = cfg

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to configured OpenAI-compatible model names."""
        # If it already looks like a provider model, pass through
        if any(
            claude_model.startswith(p) or claude_model.lower().startswith(p)
            for p in self.PASSTHROUGH_PREFIXES
        ):
            return claude_model

        # Map based on Claude model naming patterns
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.config.small_model
        elif "sonnet" in model_lower:
            return self.config.middle_model
        elif "opus" in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model


model_manager = ModelManager(config)
