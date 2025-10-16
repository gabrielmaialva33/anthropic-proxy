"""
Configuration for the application
"""
import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def parse_environment_variables():
    """Parse and validate environment variables"""
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
