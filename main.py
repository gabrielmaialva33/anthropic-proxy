import os
import sys
import uvicorn

from src.app.core.config import parse_environment_variables

# Load and validate environment variables
if not parse_environment_variables():
    sys.exit(1)

# Import app after environment variables are loaded

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python main.py [--debug]")
        print("Options:")
        print("  --debug    Run with debug logging and auto-reload")
        sys.exit(0)

    debug_mode = "--debug" in sys.argv
    log_level = "info" if debug_mode else "error"

    print(f"Starting Claude-OpenAI proxy server on http://0.0.0.0:8082")
    print(f"Log level: {log_level}")
    print(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")

    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8082,
        log_level=log_level,
        reload=debug_mode
    )
