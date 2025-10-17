#!/usr/bin/env python3
"""
Test genai-prices model matching against our commonly used models.

This test verifies that the model names we use in token_counter.py
correctly match to pricing data in the genai-prices library.
"""

from genai_prices import Usage, calc_price

# Test token count (arbitrary, just for calculating price)
TEST_TOKENS = 1000

# Models we commonly use in our token counter
TEST_MODELS = [
    # OpenAI models
    ("openai", "gpt-4o", "OpenAI GPT-4o"),
    ("openai", "gpt-4o-mini", "OpenAI GPT-4o Mini"),
    ("openai", "gpt-4", "OpenAI GPT-4"),
    ("openai", "gpt-4-turbo", "OpenAI GPT-4 Turbo"),
    ("openai", "gpt-3.5-turbo", "OpenAI GPT-3.5 Turbo"),

    # Anthropic models (from our README)
    ("anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ("anthropic", "claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
    ("anthropic", "claude-3-opus-20240229", "Claude 3 Opus"),
    ("anthropic", "claude-3-sonnet-20240229", "Claude 3 Sonnet"),
    ("anthropic", "claude-3-haiku-20240307", "Claude 3 Haiku"),

    # Gemini models
    ("google", "gemini-2.0-flash-001", "Gemini 2.0 Flash"),
    ("google", "gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("google", "gemini-1.5-pro", "Gemini 1.5 Pro"),
    ("google", "gemini-1.5-flash", "Gemini 1.5 Flash"),
]


def test_model_matching():
    """Test that all our model names can be matched by genai-prices"""
    print("Testing genai-prices model matching")
    print("=" * 80)
    print(f"Test input token count: {TEST_TOKENS}\n")

    failed = []
    mismatched = []

    for provider_id, model_ref, description in TEST_MODELS:
        try:
            price_data = calc_price(
                Usage(input_tokens=TEST_TOKENS, output_tokens=0),
                model_ref=model_ref,
                provider_id=provider_id,
            )

            # Check if the matched model ID differs from our input
            matched_id = price_data.model.id
            if matched_id != model_ref:
                mismatched.append((model_ref, matched_id))
                match_indicator = "⚠"
            else:
                match_indicator = "✓"

            # Calculate cost per 1M tokens for readability
            cost_per_1m = price_data.total_price * 1000

            print(f"{match_indicator} {description}")
            print(f"  Input model: {model_ref}")
            if matched_id != model_ref:
                print(f"  Matched to:  {matched_id}")
            print(f"  Provider:    {provider_id}")
            print(f"  Cost:        ${price_data.total_price:.6f} (${cost_per_1m:.2f}/1M tokens)")
            print()

        except Exception as e:
            failed.append((model_ref, str(e)))
            print(f"✗ {description}")
            print(f"  Model: {model_ref}")
            print(f"  Provider: {provider_id}")
            print(f"  Error: {e}")
            print()

    # Summary
    print("=" * 80)
    print(f"Results: {len(TEST_MODELS) - len(failed)} passed, {len(failed)} failed")

    if mismatched:
        print(f"\nNote: {len(mismatched)} model(s) matched to different IDs:")
        for input_model, matched_model in mismatched:
            print(f"  {input_model} → {matched_model}")
        print("This is expected behavior when genai-prices normalizes model names.")

    if failed:
        print(f"\n⚠ WARNING: {len(failed)} model(s) failed to match!")
        for model, error in failed:
            print(f"  {model}: {error}")
        return False

    return True


if __name__ == "__main__":
    success = test_model_matching()
    exit(0 if success else 1)
