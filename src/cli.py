import argparse
import json
import os
import sys

import uvicorn
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Claude-OpenAI Proxy Server")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind the server to")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info",
                        help="Logging level")

    # Mode configuration
    parser.add_argument("--mode", choices=["anthropic-to-openai", "openai-to-anthropic", "bidirectional"],
                        default="anthropic-to-openai", help="Conversion mode")

    # Advanced configuration
    parser.add_argument("--config", help="Path to configuration YAML file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--cache", action="store_true", help="Enable response caching")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")

    # Provider settings
    parser.add_argument("--openai-model", default="gpt-4o", help="Default OpenAI model to use")
    parser.add_argument("--openai-base-url", help="Custom base URL for OpenAI API")
    parser.add_argument("--anthropic-base-url", help="Custom base URL for Anthropic API")

    # Model mappings
    parser.add_argument("--map", action="append",
                        help="Model mapping in format 'source:target', can be specified multiple times")

    return parser.parse_args()


def load_config_file(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def setup_environment(args):
    """Set up environment variables from arguments"""
    # Only set env vars if not already set
    if args.openai_base_url and "OPENAI_API_BASE" not in os.environ:
        os.environ["OPENAI_API_BASE"] = args.openai_base_url

    if args.anthropic_base_url and "ANTHROPIC_API_BASE" not in os.environ:
        os.environ["ANTHROPIC_API_BASE"] = args.anthropic_base_url

    # Set default models
    if args.openai_model:
        os.environ["BIG_MODEL"] = args.openai_model

    # Handle model mappings
    if args.map:
        model_mappings = {}
        for mapping in args.map:
            if ":" in mapping:
                source, target = mapping.split(":", 1)
                model_mappings[source.strip()] = target.strip()

        if model_mappings:
            os.environ["MODEL_MAPPINGS"] = json.dumps(model_mappings)

    # Set conversion mode
    if args.mode:
        if args.mode == "anthropic-to-openai":
            os.environ["USE_OPENAI_MODELS"] = "True"
        elif args.mode == "openai-to-anthropic":
            os.environ["USE_OPENAI_MODELS"] = "False"
        elif args.mode == "bidirectional":
            os.environ["ENABLE_BIDIRECTIONAL"] = "True"

    # Set caching options
    if args.cache:
        os.environ["ENABLE_CACHE"] = "True"
        os.environ["CACHE_TTL"] = str(args.cache_ttl)


def main():
    """Main entry point for CLI"""
    args = parse_args()

    # If config file provided, load it
    if args.config:
        config = load_config_file(args.config)
        # Apply config (would merge with args)

    # Set up environment based on arguments
    setup_environment(args)

    # Print startup information
    print(f"Starting Claude-OpenAI proxy server on http://{args.host}:{args.port}")
    print(f"Mode: {args.mode}")
    print(f"Log level: {args.log_level}")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print(f"Caching: {'enabled' if args.cache else 'disabled'}")

    # Start server
    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
