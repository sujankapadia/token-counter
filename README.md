# Token Counter

A unified Python utility for counting tokens across multiple LLM providers: OpenAI, Anthropic (Claude), and Google (Gemini).

## API Key Requirements

| Provider | API Key Required? | Notes |
|----------|-------------------|-------|
| **OpenAI** | ❌ No | Uses `tiktoken` - works completely offline |
| **Gemini** | ✅ Yes | Requires API call to count tokens |
| **Anthropic** | ✅ Yes | Requires API call to count tokens |

## Features

- **Token Counting**:
  - **OpenAI**: Uses `tiktoken` library for accurate offline token counting (no API key needed)
  - **Anthropic**: Uses the Anthropic SDK's `count_tokens` API (requires API key)
  - **Gemini**: Uses the Google GenAI SDK's `count_tokens` API (requires API key)

- **Cost Estimation** (Optional):
  - Estimates input token costs using the [`genai-prices`](https://github.com/pydantic/genai-prices) library
  - Works with all three providers (OpenAI, Anthropic, Gemini)
  - Automatically matches model names to pricing data
  - Shows warnings when model names are normalized for pricing
  - Note: Only estimates input token costs (output tokens unknown at counting time)

## Installation

### Using uv (Recommended)

```bash
# Clone or navigate to the project directory
cd token-counter

# Install dependencies and create virtual environment
uv sync

# Run the CLI (option 1: using uv run)
uv run token-counter openai "Hello, world!"

# Or activate the virtual environment first (option 2)
source .venv/bin/activate
token-counter openai "Hello, world!"
```

### Using pip

If you need a requirements.txt file, generate it from the project:

```bash
# Generate requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# Then install
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install tiktoken anthropic google-genai genai-prices
```

## Usage

### Command Line Interface (Recommended)

The easiest way to count tokens is using the CLI:

```bash
# Using uv (recommended)
uv run token-counter openai "Hello, world!"
# Output: Openai (gpt-4o) (offline): 4 tokens

# Or using Python directly
python cli.py openai "Hello, world!"

# Gemini (requires API key - either GOOGLE_API_KEY or GEMINI_API_KEY)
export GOOGLE_API_KEY="your-key"  # GOOGLE_API_KEY takes precedence
uv run token-counter gemini "Hello, world!"
# Output: Gemini (gemini-2.5-flash): 4 tokens

# Anthropic (requires API key)
export ANTHROPIC_API_KEY="your-key"
uv run token-counter anthropic "Hello, world!"
# Output: Anthropic (claude-sonnet-4-20250514): 12 tokens

# Read from a file
uv run token-counter openai --file document.txt

# Read from stdin
cat document.txt | uv run token-counter openai -
echo "Hello, world!" | uv run token-counter gemini -

# Specify a model
uv run token-counter openai "Hello, world!" --model gpt-4-turbo
uv run token-counter gemini "Hello, world!" --model gemini-1.5-pro

# Quiet mode (output only the number)
uv run token-counter openai "Hello, world!" --quiet
# Output: 4

# Estimate input token cost
uv run token-counter openai "Hello, world!" --cost
# Output:
# Openai (gpt-4o) (offline): 4 tokens
# Estimated cost: $0.000010 (input tokens only)

# Cost with quiet mode (outputs: tokens,cost)
uv run token-counter anthropic "Hello, world!" --cost --quiet
# Output: 12,0.000036

# Get help
uv run token-counter --help
```

### Python API

#### OpenAI - No API Key Needed

```python
from token_counter import TokenCounter, Provider

# No API key required for OpenAI
counter = TokenCounter()

text = "Hello, world! How many tokens is this message?"

# Count tokens only
result = counter.count_tokens(text, Provider.OPENAI, "gpt-4o")
print(f"OpenAI: {result.tokens} tokens")

# Count tokens with cost estimation
result = counter.count_tokens(text, Provider.OPENAI, "gpt-4o", estimate_cost=True)
print(f"OpenAI: {result.tokens} tokens")
print(f"Estimated cost: ${result.estimated_cost:.6f}")
```

#### Anthropic & Gemini - API Keys Required

```python
from token_counter import TokenCounter, Provider
import os

# Initialize with API keys
# GOOGLE_API_KEY takes precedence over GEMINI_API_KEY for Gemini
counter = TokenCounter(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    gemini_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
)

text = "Hello, world! How many tokens is this message?"

# Anthropic with cost estimation
result = counter.count_tokens(
    text, Provider.ANTHROPIC, "claude-sonnet-4-20250514", estimate_cost=True
)
print(f"Anthropic: {result.tokens} tokens")
if result.estimated_cost:
    print(f"Estimated cost: ${result.estimated_cost:.6f}")
    # Check if model name was normalized for pricing
    if result.matched_model != "claude-sonnet-4-20250514":
        print(f"Note: Pricing matched to '{result.matched_model}'")

# Gemini
result = counter.count_tokens(
    text, Provider.GEMINI, "gemini-2.0-flash-001"
)
print(f"Gemini: {result.tokens} tokens")
```

#### Run the Example Script

```bash
# OpenAI only (no API keys needed)
python token_counter.py

# Full functionality (with API keys for Anthropic & Gemini)
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"  # or GEMINI_API_KEY
python token_counter.py
```

## Supported Models

### OpenAI
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4`, `gpt-4-turbo`
- `gpt-3.5-turbo`
- Any model supported by tiktoken

### Anthropic (Claude)
- `claude-sonnet-4-20250514`
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Google (Gemini)
- `gemini-2.0-flash-001`
- `gemini-2.5-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

## API Keys

**OpenAI**: No API key needed - works completely offline

**Anthropic & Gemini**: API keys required

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"

# For Gemini, either works (GOOGLE_API_KEY takes precedence):
export GOOGLE_API_KEY="your-google-key"
# OR
export GEMINI_API_KEY="your-gemini-key"
```

## Vertex AI Support

To use Gemini via Vertex AI instead of the Gemini API:

```python
counter = TokenCounter(
    gemini_project_id="your-gcp-project",
    gemini_location="us-central1",
    use_vertex_ai=True
)
```

## License

MIT
