"""
Helper functions for the application
"""
import logging
import sys

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """
    Log API requests in a colorful, easy-to-read format

    Args:
        method: HTTP method (GET, POST, etc)
        path: Request path
        claude_model: Original Claude model name
        openai_model: Mapped OpenAI model name
        num_messages: Number of messages in the request
        num_tools: Number of tools in the request
        status_code: HTTP response status code
    """
    # Format the Claude model name
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Get endpoint without query parameters
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Format the OpenAI model name
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    # Format tools and messages count
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code
    status_str = f"{Colors.GREEN} {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED} {status_code}{Colors.RESET}"

    # Build and print log lines
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
    sys.stdout.flush()


def format_exception(e):
    """
    Format an exception into a detailed error dictionary
    that is safely JSON serializable.

    Args:
        e: The exception object

    Returns:
        dict: Formatted error details
    """
    import traceback

    error_traceback = traceback.format_exc()
    error_details = {
        "error": str(e),
        "type": type(e).__name__,
        "traceback": error_traceback
    }

    # Extract additional attributes safely
    for attr in ['message', 'status_code', 'llm_provider', 'model']:
        if hasattr(e, attr):
            error_details[attr] = str(getattr(e, attr))  # Convert to string to ensure serializability

    # Handle response attribute specially since it's often not serializable
    if hasattr(e, 'response'):
        response = getattr(e, 'response')
        try:
            # Try to extract useful information from response
            response_info = {}
            if hasattr(response, 'status_code'):
                response_info['status_code'] = response.status_code
            if hasattr(response, 'text'):
                try:
                    response_info['text'] = response.text[:1000]  # Truncate long responses
                except:
                    pass
            if hasattr(response, 'headers'):
                try:
                    response_info['headers'] = dict(response.headers)
                except:
                    pass
            error_details['response'] = response_info
        except:
            # Fallback if we can't extract structured info
            error_details['response'] = str(response)

    # Add other attributes from __dict__, but safely
    if hasattr(e, '__dict__'):
        for key, value in e.__dict__.items():
            if key not in error_details and key not in ['args', '__traceback__']:
                try:
                    # Try to convert to string to ensure it's serializable
                    error_details[key] = str(value)
                except:
                    # If that fails, just note that we couldn't serialize it
                    error_details[key] = f"<non-serializable {type(value).__name__}>"

    return error_details


def parse_environment_variables():
    """Parse and validate environment variables"""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Check that at least one API key is present
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        logger.error("Missing API keys: At least one of ANTHROPIC_API_KEY or OPENAI_API_KEY must be provided")
        logger.error("Please set at least one of these variables in your .env file or environment.")
        return False

    # Log configuration
    logger.info("Environment variables loaded successfully")
    logger.info(f"Using big model: {os.environ.get('BIG_MODEL', 'gpt-4o')}")
    logger.info(f"Using small model: {os.environ.get('SMALL_MODEL', 'gpt-4o-mini')}")

    return True
