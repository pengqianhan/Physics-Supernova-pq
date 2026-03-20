"""Quick test: LiteLLM model → smolagents CodeAgent."""

import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from smolagents import LiteLLMModel, CodeAgent
from utils.llm_providers import get_litellm_kwargs


def test_model(model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"{'='*60}")

    llm_kwargs = get_litellm_kwargs(model_id)
    model = LiteLLMModel(
        model_id=model_id,
        **llm_kwargs,
        num_retries=3,
        timeout=60,
        thinking_level = "high" # high, low
    )

    agent = CodeAgent(tools=[], model=model, max_steps=2)
    result = agent.run("What is 2 + 3? Return the answer as a number.")
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    # Default models to test; override via command line args
    models = sys.argv[1:] or ["gemini/gemini-3-flash-preview"]

    for m in models:
        try:
            test_model(m)
        except Exception as e:
            print(f"FAILED: {m} -> {e}")
