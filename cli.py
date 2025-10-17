#!/usr/bin/env python3
"""
Command-line interface for token counting across multiple LLM providers.
"""

import sys
import argparse
from token_counter import TokenCounter, Provider


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens for text across different LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count tokens for a simple text (offline, no API key needed)
  %(prog)s openai "Hello, world!"
  %(prog)s gemini "Hello, world!"

  # Count tokens from stdin
  echo "Hello, world!" | %(prog)s openai -

  # Count tokens from a file
  %(prog)s openai --file input.txt

  # Specify a model
  %(prog)s openai "Hello, world!" --model gpt-4o
  %(prog)s gemini "Hello, world!" --model gemini-2.5-flash

  # Count tokens for Anthropic (requires API key)
  export ANTHROPIC_API_KEY="your-key"
  %(prog)s anthropic "Hello, world!"

  # Count tokens for Gemini (requires API key)
  export GOOGLE_API_KEY="your-key"  # or GEMINI_API_KEY
  %(prog)s gemini "Hello, world!"

  # Estimate cost (input tokens only)
  %(prog)s openai "Hello, world!" --cost
  %(prog)s anthropic "Hello, world!" --cost --quiet

API Key Requirements:
  OpenAI:    No API key needed (offline)
  Gemini:    Requires GOOGLE_API_KEY or GEMINI_API_KEY (GOOGLE_API_KEY takes precedence)
  Anthropic: Requires ANTHROPIC_API_KEY environment variable
        """,
    )

    parser.add_argument(
        "provider",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use for token counting",
    )

    parser.add_argument(
        "text",
        nargs="?",
        help='Text to count tokens for, or "-" to read from stdin',
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Read text from a file instead of command line",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, gemini-2.5-flash)",
    )

    parser.add_argument(
        "--anthropic-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )

    parser.add_argument(
        "--gemini-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Output only the token count number",
    )

    parser.add_argument(
        "--cost",
        action="store_true",
        help="Estimate input token cost (requires genai-prices library)",
    )

    args = parser.parse_args()

    # Get text input
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text == "-":
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either provide text, use '-' for stdin, or use --file")

    # Determine provider
    provider_map = {
        "openai": Provider.OPENAI,
        "anthropic": Provider.ANTHROPIC,
        "gemini": Provider.GEMINI,
    }
    provider = provider_map[args.provider]

    # Determine model (use defaults if not specified)
    if args.model:
        model = args.model
    else:
        default_models = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "gemini": "gemini-2.5-flash",
        }
        model = default_models[args.provider]

    # Get API keys from args or environment
    import os

    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    # GOOGLE_API_KEY takes precedence over GEMINI_API_KEY
    gemini_key = args.gemini_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Initialize counter
    counter = TokenCounter(
        anthropic_api_key=anthropic_key,
        gemini_api_key=gemini_key,
    )

    # Count tokens
    try:
        result = counter.count_tokens(text, provider, model, estimate_cost=args.cost)

        if args.quiet:
            if args.cost and result.estimated_cost is not None:
                print(f"{result.tokens},{result.estimated_cost:.6f}")
            else:
                print(result.tokens)
        else:
            mode = " (offline)" if provider == Provider.OPENAI else ""
            print(f"{args.provider.capitalize()} ({model}){mode}: {result.tokens} tokens")

            if args.cost:
                if result.estimated_cost is not None:
                    print(f"Estimated cost: ${result.estimated_cost:.6f} (input tokens only)")
                    # Show warning if model was matched to a different ID
                    if result.matched_model and result.matched_model != model:
                        print(f"Note: Pricing matched to model '{result.matched_model}'")
                else:
                    print("Cost estimation unavailable (pricing data not found)")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
