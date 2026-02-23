import base64
import os
from pathlib import Path
from typing import Any

from litellm import completion
from smolagents.default_tools import Tool


class LocalImageQATool(Tool):
    """
    Minimal image QA tool backed by LiteLLM.
    Input: local image path + question
    Output: model answer about the image
    """

    name = "local_image_qa"
    description = (
        "Answer a question about a local image file. "
        "Use this when you generated plots (png/jpg/webp/gif) and need vision feedback."
    )
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Local image path, e.g. duffing_data_preview.png",
        },
        "question": {
            "type": "string",
            "description": "Question about the image content.",
        },
    }
    output_type = "string"

    def __init__(self, model_id: str = "gemini/gemini-flash-lite-latest"):
        super().__init__()
        self.model_id = model_id

    def _get_litellm_kwargs(self) -> dict[str, Any]:
        """Mirror model routing logic used in run_llm_ha_gamma.py."""
        if not self.model_id:
            return {}
        if (
            self.model_id.startswith("moonshot/")
            or self.model_id.startswith("openai/kimi")
            or self.model_id.startswith("openai/moonshot")
            or self.model_id.startswith("kimi")
        ):
            return {
                "api_key": os.environ.get("MOONSHOT_API_KEY"),
                "api_base": os.environ.get("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
            }
        if self.model_id.startswith("gemini/"):
            return {"api_key": os.environ.get("GEMINI_API_KEY")}
        return {}

    @staticmethod
    def _guess_mime_type(image_path: str) -> str:
        ext = os.path.splitext(image_path.lower())[1]
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/png")

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            return "\n".join(texts).strip()
        return ""

    def _resolve_image_path(self, image_path: str) -> str | None:
        """
        Resolve image path with fallback lookup in evaluation result directories.

        Resolution order:
        1) As-provided path (absolute or relative to current working directory)
        2) Relative path under evaluation_results/
        3) Filename lookup under:
           - evaluation_results/**/iter_*/
           - evaluation_results/**/iter_*/temp/**
           - evaluation_results/**/best_iter_*/
           - evaluation_results/**/best_iter_*/temp/**
        """
        cleaned = image_path.strip()
        if not cleaned:
            return None

        # Direct checks first.
        direct_path = Path(os.path.expanduser(cleaned))
        if direct_path.is_file():
            return str(direct_path.resolve())
        if not direct_path.is_absolute():
            cwd_candidate = (Path.cwd() / direct_path)
            if cwd_candidate.is_file():
                return str(cwd_candidate.resolve())

        eval_root = Path.cwd() / "evaluation_results"
        if not eval_root.is_dir():
            return None

        # If caller omitted evaluation_results prefix, try resolving under it directly.
        if not direct_path.is_absolute():
            under_eval = eval_root / direct_path
            if under_eval.is_file():
                return str(under_eval.resolve())

        # Fallback: search by basename in iter folders.
        basename = direct_path.name
        if not basename:
            return None

        candidates: list[Path] = []
        patterns = [
            f"**/iter_*/{basename}",
            f"**/iter_*/temp/{basename}",
            f"**/iter_*/temp/**/{basename}",
            f"**/best_iter_*/{basename}",
            f"**/best_iter_*/temp/{basename}",
            f"**/best_iter_*/temp/**/{basename}",
        ]
        for pattern in patterns:
            candidates.extend([p for p in eval_root.glob(pattern) if p.is_file()])

        if not candidates:
            return None

        # Prefer latest artifact when multiple runs have same filename (e.g., overlay_0.png).
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())

    def forward(self, image_path: str, question: str) -> str:  # type: ignore[override]
        resolved_path = self._resolve_image_path(image_path)
        if not resolved_path:
            return (
                f"Error: image file not found: {image_path}. "
                "Also searched under evaluation_results/**/iter_*/ and temp subfolders."
            )
        abs_path = resolved_path

        try:
            with open(abs_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            return f"Error: failed to read image file: {e}"

        if not image_bytes:
            return f"Error: image file is empty: {image_path}"

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = self._guess_mime_type(abs_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ]

        try:
            response = completion(
                model=self.model_id,
                messages=messages,
                **self._get_litellm_kwargs(),
            )
            content = response.choices[0].message.content
            text = self._extract_text_content(content)
            if text:
                return text
            return "No textual response from vision model."
        except Exception as e:
            return f"Error: vision request failed: {e}"
