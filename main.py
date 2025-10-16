"""
Anthropic-OpenAI Proxy Server Entrypoint

This is the main entrypoint for the proxy server. It initializes configuration,
sets up logging, and starts the uvicorn server.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import uvicorn
from src.core.config import config
from src.core.logging import setup_logging


def main():
    """Main entry point for the proxy server."""
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Anthropic-OpenAI Proxy Server v2.0.0")
        print("")
        print("Usage: python main.py [--debug]")
        print("")
        print("Options:")
        print("  --debug    Run with debug logging and auto-reload")
        print("")
        print("Environment Variables:")
        print("  ANTHROPIC_API_KEY     - Anthropic API key")
        print("  OPENAI_API_KEY        - OpenAI API key")
        print("  NVIDIA_NIM_API_KEY    - NVIDIA NIM API key")
        print("  PREFERRED_PROVIDER    - Preferred provider (openai, anthropic, nvidia)")
        print("  BIG_MODEL             - Model for large requests")
        print("  SMALL_MODEL           - Model for small requests")
        print("  OPENAI_BASE_URL       - Custom OpenAI base URL")
        print("  SERVER_HOST           - Server host (default: 0.0.0.0)")
        print("  SERVER_PORT           - Server port (default: 8082)")
        print("  LOG_LEVEL             - Log level (default: error)")
        print("  MAX_TOKENS_LIMIT      - Maximum tokens limit (default: 16384)")
        print("  REQUEST_TIMEOUT       - Request timeout in seconds (default: 90)")
        print("")
        sys.exit(0)

    debug_mode = "--debug" in sys.argv

    # Set up logging based on debug mode
    log_level = "DEBUG" if debug_mode else config.log_level
    setup_logging(log_level)

    # Print configuration
    print("")
    config.print_configuration()

    print(f"🚀 Starting server on http://{config.server_host}:{config.server_port}")
    print(f"📊 Log level: {log_level}")
    print(f"🔧 Debug mode: {'enabled' if debug_mode else 'disabled'}")
    print(f"🔄 Auto-reload: {'enabled' if debug_mode else 'disabled'}")
    print("")

    # Start uvicorn server
    uvicorn.run(
        "src.app.main:app",
        host=config.server_host,
        port=config.server_port,
        log_level=log_level.lower(),
        reload=debug_mode
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Failed to start server: {e}")
        sys.exit(1)
