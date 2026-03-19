import os


def get_litellm_kwargs(model_id: str) -> dict:
    """
    Return the appropriate api_key and api_base for a given model_id.

    Supports:
    - moonshot/*, openai/kimi*: Moonshot API (MOONSHOT_API_KEY)
    - gemini/*: Google Gemini API (GEMINI_API_KEY)
    - Other providers: rely on litellm's env-var auto-detection
    """
    if not model_id:
        return {}

    if (
        model_id.startswith("moonshot/")
        or model_id.startswith("openai/kimi")
        or model_id.startswith("openai/moonshot")
        or model_id.startswith("kimi")
    ):
        return {
            "api_key": os.environ.get("MOONSHOT_API_KEY"),
            "api_base": os.environ.get("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
        }
    elif model_id.startswith("gemini/"):
        return {
            "api_key": os.environ.get("GEMINI_API_KEY"),
        }
    else:
        # For other providers (openrouter, anthropic, etc.), let litellm handle it
        return {}


def is_native_gemini(model_id: str) -> bool:
    """Check if model_id refers to a Gemini model (requires 'gemini/' prefix).

    Gemini models use the google-genai SDK directly to support features like
    code_execution. The 'gemini/' prefix is required; bare names like
    'gemini-3-flash-preview' will raise ValueError.
    """
    if model_id.startswith("gemini") and not model_id.startswith("gemini/"):
        raise ValueError(
            f"Invalid model_id '{model_id}'. Gemini models must use the "
            f"'gemini/' prefix, e.g. 'gemini/{model_id}'."
        )
    return model_id.startswith("gemini/")


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from environment."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
