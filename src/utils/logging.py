"""
Logging configuration for the application
"""
import logging
import sys


class MessageFilter(logging.Filter):
    """
    Filter to remove specific log messages that are noisy
    """

    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


class ColorizedFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log messages
    """
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg:
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"

        log_level_colors = {
            logging.DEBUG: self.BLUE,
            logging.INFO: self.GREEN,
            logging.WARNING: self.YELLOW,
            logging.ERROR: self.RED,
            logging.CRITICAL: f"{self.BOLD}{self.RED}",
        }

        level_color = log_level_colors.get(record.levelno, self.RESET)
        formatted_message = super().format(record)
        return f"{level_color}{formatted_message}{self.RESET}"


def setup_logging(level=logging.WARN):
    """
    Configure application logging

    Args:
        level: The logging level to use (default: WARN)
    """
    # Reset any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure basic logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set up specific loggers
    logger = logging.getLogger(__name__.split('.')[0])  # Get the package logger

    # Apply message filter to suppress noisy messages
    message_filter = MessageFilter()
    root_logger.addFilter(message_filter)

    # Apply colorized formatter to console handler
    for handler in logger.handlers + root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Set lower log levels for certain modules
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    return logger
