"""
Token Counter Utility for Multiple LLM Providers

Supports token counting for:
- OpenAI models (via tiktoken) - OFFLINE, no API key needed
- Anthropic Claude models (via Anthropic SDK) - requires API key
- Google Gemini models (via Google GenAI SDK) - requires API key
"""

from typing import Optional, Dict, Any
from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class TokenCounter:
    """Unified token counter for multiple LLM providers"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_project_id: Optional[str] = None,
        gemini_location: Optional[str] = None,
        use_vertex_ai: bool = False
    ):
        """
        Initialize the token counter.

        Args:
            anthropic_api_key: API key for Anthropic (required for Claude token counting)
            gemini_api_key: API key for Gemini (required for Gemini token counting)
            gemini_project_id: GCP project ID for Vertex AI
            gemini_location: GCP location for Vertex AI (e.g., 'us-central1')
            use_vertex_ai: Whether to use Vertex AI for Gemini (default: False)
        """
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.gemini_project_id = gemini_project_id
        self.gemini_location = gemini_location
        self.use_vertex_ai = use_vertex_ai

        # Lazy-loaded clients
        self._anthropic_client = None
        self._gemini_client = None
        self._tiktoken_cache: Dict[str, Any] = {}

    def count_tokens(
        self,
        text: str,
        provider: Provider,
        model: str
    ) -> int:
        """
        Count tokens for the given text using the specified provider and model.

        Args:
            text: The text to count tokens for
            provider: The LLM provider (OpenAI, Anthropic, or Gemini)
            model: The specific model name (e.g., 'gpt-4', 'claude-3-opus-20240229', 'gemini-2.0-flash-001')

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If the provider is not supported or API key is missing
            ImportError: If required SDK is not installed

        Notes:
            - OpenAI: Always offline, no API key needed
            - Anthropic: Requires API key (makes API call)
            - Gemini: Requires API key (makes API call)
        """
        if provider == Provider.OPENAI:
            return self._count_openai_tokens(text, model)
        elif provider == Provider.ANTHROPIC:
            return self._count_anthropic_tokens(text, model)
        elif provider == Provider.GEMINI:
            return self._count_gemini_tokens(text, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens using tiktoken for OpenAI models"""
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for OpenAI token counting. "
                "Install it with: pip install tiktoken"
            )

        # Cache encodings for efficiency
        if model not in self._tiktoken_cache:
            try:
                # Try to get encoding for the specific model
                self._tiktoken_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to a default encoding if model not recognized
                # o200k_base is used for newer models like GPT-4o
                self._tiktoken_cache[model] = tiktoken.get_encoding("o200k_base")

        encoding = self._tiktoken_cache[model]
        tokens = encoding.encode(text)
        return len(tokens)

    def _count_anthropic_tokens(self, text: str, model: str) -> int:
        """Count tokens using Anthropic SDK"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK is required for Claude token counting. "
                "Install it with: pip install anthropic"
            )

        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=self.anthropic_api_key)

        # Use the count_tokens API
        result = self._anthropic_client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}]
        )

        return result.input_tokens

    def _count_gemini_tokens(self, text: str, model: str) -> int:
        """Count tokens using Google GenAI SDK (requires API key)"""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai SDK is required for Gemini token counting. "
                "Install it with: pip install google-genai"
            )

        # Initialize client if needed
        if self._gemini_client is None:
            if self.use_vertex_ai:
                # Use Vertex AI
                if not self.gemini_project_id:
                    raise ValueError(
                        "gemini_project_id is required when use_vertex_ai=True"
                    )
                self._gemini_client = genai.Client(
                    vertexai=True,
                    project=self.gemini_project_id,
                    location=self.gemini_location or "us-central1"
                )
            else:
                # Use Gemini API
                if not self.gemini_api_key:
                    raise ValueError(
                        "Gemini API key is required for token counting. "
                        "Provide gemini_api_key or set GEMINI_API_KEY environment variable."
                    )
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)

        # Use the count_tokens API
        response = self._gemini_client.models.count_tokens(
            model=model,
            contents=text
        )

        return response.total_tokens


def main():
    """Example usage of the TokenCounter"""
    import os

    # Example text
    text = "Hello, world! How many tokens is this message?"

    print("Token counts for:", repr(text))
    print()

    # Example 1: OpenAI (no API key needed)
    print("=== OpenAI (Offline - No API key needed) ===")
    counter = TokenCounter()

    try:
        openai_tokens = counter.count_tokens(text, Provider.OPENAI, "gpt-4o")
        print(f"OpenAI (gpt-4o): {openai_tokens} tokens ✓")
    except Exception as e:
        print(f"OpenAI error: {e}")

    print()
    print("=== Anthropic & Gemini (API keys required) ===")

    # Example 2: With API keys
    # GOOGLE_API_KEY takes precedence over GEMINI_API_KEY
    counter_with_keys = TokenCounter(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )

    # Anthropic
    try:
        anthropic_tokens = counter_with_keys.count_tokens(
            text, Provider.ANTHROPIC, "claude-sonnet-4-20250514"
        )
        print(f"Anthropic (claude-sonnet-4): {anthropic_tokens} tokens")
    except Exception as e:
        print(f"Anthropic error: {e}")

    # Gemini
    try:
        gemini_tokens = counter_with_keys.count_tokens(
            text, Provider.GEMINI, "gemini-2.0-flash-001"
        )
        print(f"Gemini (gemini-2.0-flash): {gemini_tokens} tokens")
    except Exception as e:
        print(f"Gemini error: {e}")


if __name__ == "__main__":
    main()
