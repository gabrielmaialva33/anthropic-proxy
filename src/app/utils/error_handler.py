"""
Error handling functions for the application
"""


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
