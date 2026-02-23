"""
最小示例：展示 smolagents 读取本地图片的能力。

官方文档关键点（已按官方示例实现）：
1) MultiStepAgent.run 支持 images 参数：
   agent.run("Describe these images", images=[image_1, image_2])
2) 传入的图片会作为任务图像输入给模型。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from smolagents import CodeAgent, LiteLLMModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal smolagents image reading demo.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="analysis_plots_train/non_linear/duffing/sample_0_analysis.png",
        help="Path to a local image file.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Vision-capable model id routed by LiteLLM.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "请阅读这张图并给出："
            "1) 图像主要内容；"
            "2) 你看到的关键趋势；"
            "3) 一句简短结论。"
        ),
        help="Task prompt passed to the agent.",
    )
    return parser


def main() -> None:
    load_dotenv(override=True)
    args = build_parser().parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment or .env")

    model = LiteLLMModel(
        model_id=args.model_id,
        api_key=gemini_api_key,
        max_completion_tokens=1024,
    )
    agent = CodeAgent(
        model=model,
        tools=[],
        add_base_tools=False,
        max_steps=3,
        verbosity_level=1,
    )

    with Image.open(image_path) as img:
        image_for_agent = img.convert("RGB").copy()

    print(f"[demo] image: {image_path}")
    print(f"[demo] model: {args.model_id}")
    print("[demo] running agent with images=[PIL.Image] ...\n")
    result = agent.run(args.prompt, images=[image_for_agent])

    print("===== Agent Response =====")
    print(result)


if __name__ == "__main__":
    main()
