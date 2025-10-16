# Gemini Code Assistant

This document provides an overview of the `anthropic-proxy` project, its architecture, and development conventions to be used as a context for Gemini.

## Project Overview

`anthropic-proxy` is a Python-based proxy server that acts as a bridge between the Anthropic and OpenAI API ecosystems. It allows developers to use Anthropic's client-side tools and libraries (like Claude Code) with OpenAI's powerful language models (e.g., GPT-4o, GPT-4o-mini).

The core functionality of the proxy is to:

1.  **Receive** API requests in Anthropic's format.
2.  **Translate** these requests into the corresponding OpenAI API format.
3.  **Forward** the translated requests to the OpenAI API.
4.  **Receive** the response from OpenAI.
5.  **Convert** the OpenAI response back into Anthropic's format.
6.  **Return** the final response to the client.

This process is designed to be seamless and transparent, handling various complexities such as streaming responses, tool calls, system prompts, and multi-turn conversations.

### Key Technologies

*   **Python 3.10+**: The core programming language.
*   **FastAPI**: A modern, high-performance web framework for building the API.
*   **LiteLLM**: A library that provides a unified interface for interacting with various LLM APIs, handling the core translation logic.
*   **Pydantic**: Used for data validation and settings management, ensuring that API requests and responses adhere to the expected schemas.
*   **Uvicorn**: An ASGI server that runs the FastAPI application.
*   **python-dotenv**: Manages environment variables for configuration.

### Architecture

The project follows a standard Python application structure:

*   `main.py`: The main entry point for starting the server.
*   `src/app/`: Contains the core application logic.
    *   `main.py`: FastAPI application setup, middleware, and exception handlers.
    *   `api/routes.py`: Defines the API endpoints, including `/v1/messages` for handling chat completions.
    *   `core/config.py`: Manages application configuration and environment variables.
    *   `models/schema.py`: Pydantic models that define the data structures for API requests and responses.
    *   `services/converter.py`: The heart of the proxy, containing the logic for converting data between Anthropic and LiteLLM/OpenAI formats.
    *   `utils/`: Utility modules for logging, error handling, and other helper functions.
*   `tests/`: Contains unit and integration tests for the application.
    *   `test_api.py`: Integration tests that send requests to the running proxy and compare its output to the native Anthropic API.
    *   `test_converter.py`: Unit tests for the conversion logic in `converter.py`.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `pyproject.toml`: Project metadata and build configuration.
*   `Dockerfile` and `docker-compose.yml`: For containerizing the application.

## Building and Running

### Prerequisites

*   Python 3.10+
*   `uv` or `pip` for package management
*   An OpenAI API key

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/gabrielmaialva33/anthropic-proxy.git
    cd anthropic-proxy
    ```

2.  **Install dependencies:**
    ```sh
    # With uv (recommended)
    uv pip install -r requirements.txt

    # Or with pip
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    Create a `.env` file in the project root and add the following:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    BIG_MODEL=gpt-4o
    SMALL_MODEL=gpt-4o-mini
    ```

### Running the Server

You can run the server in two ways:

1.  **Using the main script (recommended for development):**
    ```sh
    python main.py
    ```
    This will start the server on `http://0.0.0.0:8082`.

2.  **Using Uvicorn directly:**
    ```sh
    uvicorn src.app.main:app --host 0.0.0.0 --port 8082
    ```

### Running Tests

The project has a suite of tests to ensure that the proxy is functioning correctly.

*   **Run all API tests:**
    ```sh
    python -m tests.test_api
    ```

*   **Run converter unit tests:**
    ```sh
    python -m tests.test_converter
    ```

## Development Conventions

### Coding Style

*   The project follows the **PEP 8** style guide for Python code.
*   **Type hints** are used throughout the codebase to improve readability and allow for static analysis.
*   Code is organized into logical modules with clear responsibilities.

### Testing

*   The `tests` directory contains both integration tests (`test_api.py`) and unit tests (`test_converter.py`).
*   `test_api.py` is particularly important as it performs end-to-end tests by sending requests to both the proxy and the official Anthropic API and comparing the results. This ensures that the proxy's output is consistent with the native API.
*   When adding new features or fixing bugs, it is expected that corresponding tests will be added or updated to maintain coverage and prevent regressions.

### Dependency Management

*   Project dependencies are managed in `requirements.txt`.
*   `uv` is the recommended tool for installing and managing dependencies due to its speed.

### Commits and Versioning

*   Commit messages should be clear and descriptive, explaining the "what" and "why" of the changes.
*   The project uses semantic versioning, although it is still in its early stages (v0.1.0).

### Logging

*   The application uses Python's built-in `logging` module.
*   Logging is configured in `src/app/utils/logging.py`.
*   The log level can be configured via the `LOG_LEVEL` environment variable.

This `GEMINI.md` file should provide a solid foundation for understanding the `anthropic-proxy` project.
