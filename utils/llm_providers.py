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