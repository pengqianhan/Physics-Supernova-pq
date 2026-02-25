import argparse
import base64
import mimetypes
import os
from pathlib import Path

from smolagents import LiteLLMModel
from dotenv import load_dotenv

load_dotenv()


def image_path_to_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def main() -> None:
    default_image = Path(__file__).resolve().parents[1] / "input_signals_detailed.png"

    parser = argparse.ArgumentParser(description="Test kimi-k2.5 image understanding with LiteLLM.")
    parser.add_argument("--image-path", type=Path, default=default_image, help="Path to image file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="请详细描述这张图中的内容，并说明它可能反映了什么系统行为。",
        help="Question prompt for the model",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise EnvironmentError("MOONSHOT_API_KEY is not set. Please configure it in your environment or .env.")

    model = LiteLLMModel(
        model_id="openai/kimi-k2.5",
        api_key=api_key,
        api_base="https://api.moonshot.cn/v1",
    )

    image_data_url = image_path_to_data_url(args.image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]

    response = model(messages=messages)
    print(response)


if __name__ == "__main__":
    main()