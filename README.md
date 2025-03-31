<table style="width:100%" align="center" border="0">
  <tr align="center">
    <td><img src="assets/logo.png" alt="Claude on OpenAI" width="300"></td>
    <td><h1>🧩 Claude on OpenAI 🔄</h1></td>
  </tr>
</table>

<p align="center">
  <strong>A proxy that lets you use Claude Code with OpenAI models like GPT-4o.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/gabrielmaialva33/anthropic-proxy?color=00b8d3?style=flat&logo=appveyor" alt="License" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat&logo=python" alt="Python" >
  <img src="https://img.shields.io/badge/FastAPI-0.115.11-009688.svg?style=flat&logo=fastapi" alt="FastAPI" >
  <img src="https://img.shields.io/badge/OpenAI-API-412991.svg?style=flat&logo=openai" alt="OpenAI" >
  <img src="https://img.shields.io/badge/Claude-Code-5A67D8?style=flat&logo=anthropic" alt="Claude Code" >
  <img src="https://img.shields.io/badge/made%20by-Maia-15c3d6?style=flat&logo=appveyor" alt="Maia" >  
</p>

<br>

<p align="center">
  <a href="#bookmark-about">About</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#computer-technologies">Technologies</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#package-installation">Installation</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#rocket-usage">Usage</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#gear-how-it-works">How it Works</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#memo-license">License</a>
</p>

<br>

## :bookmark: About

**Claude on OpenAI** is a proxy server that lets you use Claude Code (Anthropic's CLI tool) with OpenAI models like
GPT-4o and GPT-4o-mini. It translates between Anthropic's API format and OpenAI's API format, allowing you to leverage
OpenAI's powerful models through Anthropic's interfaces.

<br>

## :computer: Technologies

- **[Python](https://python.org/)** - Core language
- **[FastAPI](https://fastapi.tiangolo.com/)** - Web framework
- **[LiteLLM](https://github.com/BerriAI/litellm)** - API translation layer
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server
- **[dotenv](https://pypi.org/project/python-dotenv/)** - Environment variable management

<br>

## :package: Installation

### :gear: **Prerequisites**

- **[Python](https://python.org/)** (3.10+)
- **[uv](https://pypi.org/project/uv/)** or **[pip](https://pypi.org/project/pip/)** for package management
- An **OpenAI API key**
- Optionally, an **Anthropic API key** if you want to use Anthropic models too

<br>

### :octocat: **Cloning the repository**

```sh
git clone https://github.com/gabrielmaialva33/anthropic-proxy.git
cd anthropic-proxy
```

<br>

### :whale: **Installing dependencies**

```sh
# With uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

<br>

### :key: **Configuration**

Create a `.env` file with:

```env
# API Keys (at least one is required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Required for Anthropic models
OPENAI_API_KEY=your_openai_api_key_here        # Required for OpenAI models

# Model Configuration
BIG_MODEL=gpt-4o                  # OpenAI model to use for Claude Sonnet models
SMALL_MODEL=gpt-4o-mini           # OpenAI model to use for Claude Haiku models

# Feature Flags
USE_OPENAI_MODELS=True            # Set to False to use native Anthropic models instead

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8082
LOG_LEVEL=error                   # Options: debug, info, warning, error, critical
```

<br>

## :rocket: Usage

### :computer: **Starting the proxy server**

```sh
# With Python
python main.py

# With uvicorn directly
uvicorn src.app:app --host 0.0.0.0 --port 8082
```

<br>

### :shell: **Using with Claude Code**

1. **Install Claude Code** (if not already installed):
   ```sh
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```sh
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. That's it! Your Claude Code client will now use OpenAI models through the proxy.

<br>

## :gear: How it Works

This proxy operates by:

1. **Receiving requests** in Anthropic's API format
2. **Translating** the requests to OpenAI format using LiteLLM
3. **Sending** the translated request to OpenAI
4. **Converting** the response back to Anthropic format
5. **Returning** the formatted response to the client

The proxy handles both streaming and non-streaming responses, tool calls, system prompts, and multi-turn conversations
to maintain compatibility with all Claude clients.

<br>

### :world_map: Model Mapping

The proxy automatically maps Claude models to OpenAI models:

| Claude Model | OpenAI Model          |
|--------------|-----------------------|
| haiku        | gpt-4o-mini (default) |
| sonnet       | gpt-4o (default)      |

<br>

You can customize which OpenAI models are used via the `BIG_MODEL` and `SMALL_MODEL` environment variables.

<br>

## :test_tube: Running Tests

```sh
# Run all tests
python -m tests.test_api

# Run only non-streaming tests
python -m tests.test_api --no-streaming

# Run only simple tests (no tools)
python -m tests.test_api --simple

# Run only tool tests
python -m tests.test_api --tools-only

# Run unit tests for the converter
python -m tests.test_converter
```

<br>

## :memo: License

This project is under the **MIT** license. [MIT](./LICENSE) ❤️

<br>

## :rocket: **Contributors**

| [![Maia](https://avatars.githubusercontent.com/u/26732067?size=100)](https://github.com/gabrielmaialva33) |
|-----------------------------------------------------------------------------------------------------------|
| [Maia](https://github.com/gabrielmaialva33)                                                               |

Made with ❤️ by Maia 👋🏽 [Get in touch!](https://t.me/mrootx)

## :star:

Liked it? Leave a little star to help the project ⭐

<br/>

<p align="center"><img src="https://raw.githubusercontent.com/gabrielmaialva33/gabrielmaialva33/master/assets/gray0_ctp_on_line.svg?sanitize=true" /></p>
<p align="center">&copy; 2024-present <a href="https://github.com/gabrielmaialva33/" target="_blank">Maia</a></p>