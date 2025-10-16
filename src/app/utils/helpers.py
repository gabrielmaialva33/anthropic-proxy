"""
Helper functions for the application
"""
import sys

import logging

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
