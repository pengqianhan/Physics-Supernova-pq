import os
import time
import base64
from PIL import Image
from io import BytesIO
from smolagents.default_tools import Tool
from base64 import b64decode

from .markdown_utils import MarkdownMessage
from .llm_providers import get_litellm_kwargs, is_native_gemini, get_gemini_api_key
from dotenv import load_dotenv
load_dotenv()


class HybridAutomatonImageTool(Tool):
    """Image expert to analyse the plot of the data collected from the system. You MUST call this tool when you need to measure quantities from an image."""

    name = "hybrid_automaton_image_analysis"
    description = (
        "An advanced image expert tool equipped with a Python Code Execution Environment."
        "Given an image placeholder and a question, return the image expert's answer about that image. "
        "Accepts placeholders only: (1) <image_N> for original trace images from markdown, or "
        "(2) <iter_image_X_Y> for evaluator plots registered during iteration. "
        "When you need to measure quantities or analyze trajectory comparisons, you MUST call this tool."
    )
    inputs = {
        "image_ref": {
            "type": "string",
            "description": "Image placeholder: <image_N> for trace images or <iter_image_X_Y> for evaluator plots"
        },
        "question": {"type": "string", "description": "Question to ask about the image"},
    }
    output_type = "string"

    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    ALLOWED_PATH_PREFIXES = ['evaluation_results', 'data_all/']

    def __init__(self, worker_agent=None, vision_model_id: str = "gemini/gemini-3-flash-preview", max_short_side_pixels: int = 9999):
        super().__init__()
        self.worker_agent = worker_agent
        self.vision_model_id = vision_model_id
        self.max_short_side_pixels = max_short_side_pixels
        self._iteration_images: dict[int, bytes] = {}

        # Initialize native Gemini client only when needed
        self._gemini_client = None
        if is_native_gemini(vision_model_id):
            from google import genai
            self._gemini_client = genai.Client(api_key=get_gemini_api_key())

    # ── Image registration & extraction ────────────────────────────────

    def _is_valid_file_path(self, path: str) -> tuple[bool, str]:
        if '..' in path:
            return False, "Path traversal (..) is not allowed for security reasons."
        _, ext = os.path.splitext(path.lower())
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File extension '{ext}' not allowed. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
        path_normalized = path.replace('\\', '/')
        if not any(path_normalized.startswith(prefix) for prefix in self.ALLOWED_PATH_PREFIXES):
            abs_path = os.path.abspath(path)
            cwd = os.getcwd()
            for prefix in self.ALLOWED_PATH_PREFIXES:
                allowed_abs = os.path.abspath(os.path.join(cwd, prefix))
                if abs_path.startswith(allowed_abs):
                    return True, ""
            return False, f"Path must start with one of: {', '.join(self.ALLOWED_PATH_PREFIXES)}"
        return True, ""

    def _load_image_bytes_from_file(self, file_path: str) -> tuple[bytes | None, str]:
        is_valid, error_msg = self._is_valid_file_path(file_path)
        if not is_valid:
            return None, error_msg
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return None, f"File not found: {file_path}"
        try:
            with open(abs_path, 'rb') as f:
                return f.read(), ""
        except Exception as e:
            return None, f"Failed to read file: {str(e)}"

    def register_iteration_image(self, index: int, source: str | bytes) -> tuple[bool, str]:
        """Register an evaluator plot image for later analysis via <iter_image_X_Y>.

        Use 100 * X + Y as the index.
        """
        if isinstance(source, bytes):
            self._iteration_images[index] = source
            return True, ""
        elif isinstance(source, str):
            img_bytes, error_msg = self._load_image_bytes_from_file(source)
            if img_bytes is None:
                return False, error_msg
            self._iteration_images[index] = img_bytes
            return True, ""
        return False, f"Invalid source type: {type(source)}. Expected bytes or str (file path)."  # type: ignore[unreachable]

    def clear_iteration_images(self) -> None:
        self._iteration_images.clear()

    def get_registered_iteration_indices(self) -> list[int]:
        return sorted(self._iteration_images.keys())

    def _extract_image_bytes(self, image_ref: str) -> tuple[bytes | None, str]:
        """Extract image bytes from a placeholder reference (<image_N> or <iter_image_X_Y>)."""
        # Handle <iter_image_N> or <iter_image_X_Y>
        if image_ref.startswith("<iter_image_") and image_ref.endswith(">"):
            inner = image_ref[len("<iter_image_"):-1]
            try:
                if '_' in inner:
                    parts = inner.split('_')
                    idx = int(parts[0]) * 100 + int(parts[1])
                else:
                    idx = int(inner)
            except (ValueError, IndexError):
                return None, f"Invalid iteration image placeholder format: {image_ref}"
            if idx not in self._iteration_images:
                available = self.get_registered_iteration_indices()
                if available:
                    return None, f"Iteration image {idx} not registered. Available: {available}"
                return None, "No iteration images registered. Call register_iteration_image() first."
            return self._iteration_images[idx], ""

        # Handle <image_N>
        if image_ref.startswith("<image_") and image_ref.endswith(">"):
            try:
                idx = int(image_ref[len("<image_"):-1])
            except ValueError:
                return None, f"Invalid image placeholder format: {image_ref}"
        elif image_ref.isdigit():
            idx = int(image_ref)
        else:
            return None, f"Invalid placeholder format: {image_ref}. Use <image_N> or <iter_image_N>."

        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            return None, "No markdown content available. Ensure worker_agent is set with markdown_content_high_res_image."
        md: MarkdownMessage = self.worker_agent.markdown_content_high_res_image
        img_blocks = [it for it in md.content if it.get("type") == "image_url"]
        if not (0 <= idx < len(img_blocks)):
            return None, f"Image index {idx} out of range. Available: 0-{len(img_blocks) - 1}"
        data_url = img_blocks[idx]["image_url"]["url"]
        if data_url.startswith("data:image"):
            base64_part = data_url.split(",", 1)[1]
            return b64decode(base64_part), ""
        return None, f"Image {idx} has invalid data URL format."

    # ── Resize helper ──────────────────────────────────────────────────

    def _resize_if_needed(self, img_bytes: bytes) -> tuple[Image.Image, bytes]:
        """Resize image if its short side exceeds max_short_side_pixels.

        Returns both the PIL Image and the (possibly re-encoded) bytes.
        """
        img = Image.open(BytesIO(img_bytes))
        width, height = img.size
        if self.max_short_side_pixels and min(width, height) > self.max_short_side_pixels:
            scale = self.max_short_side_pixels / min(width, height)
            img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        return img, img_bytes

    # ── Model backends ─────────────────────────────────────────────────

    def _forward_gemini(self, img: Image.Image, full_prompt: str) -> str:
        """Call Gemini via native genai.Client (supports code_execution)."""
        from google.genai import types

        model_name = self.vision_model_id
        if model_name.startswith("gemini/"):
            model_name = model_name[len("gemini/"):]

        for _ in range(3):
            try:
                response = self._gemini_client.models.generate_content(
                    model=model_name,
                    contents=[img, full_prompt],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]),
                )
                parts_text = []
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        parts_text.append(part.text)
                    if part.executable_code is not None and part.executable_code.code is not None:
                        parts_text.append(part.executable_code.code)
                    if part.code_execution_result is not None and part.code_execution_result.output is not None:
                        parts_text.append(part.code_execution_result.output)
                result = '\n'.join(parts_text).strip()
                if result:
                    return result
            except Exception as e:
                print(f"Error during Gemini vision generation: {e}")
                time.sleep(5)
        return "No response"

    def _forward_litellm(self, img_bytes: bytes, full_prompt: str) -> str:
        """Call OpenAI-compatible vision models via litellm."""
        from litellm import completion as litellm_completion

        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        # Guess MIME type from magic bytes
        if img_bytes[:3] == b'\xff\xd8\xff':
            mime_type = "image/jpeg"
        elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            }
        ]

        for _ in range(3):
            try:
                response = litellm_completion(
                    model=self.vision_model_id,
                    messages=messages,
                    **get_litellm_kwargs(self.vision_model_id),
                )
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    joined = "\n".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ).strip()
                    if joined:
                        return joined
            except Exception as e:
                print(f"Error during litellm vision generation: {e}")
                time.sleep(5)
        return "No response"

    # ── Main entry point ───────────────────────────────────────────────

    def forward(self, image_ref: str, question: str) -> str:  # type: ignore[override]
        """Process image analysis request and return expert response."""
        img_bytes, error_msg = self._extract_image_bytes(image_ref)
        if img_bytes is None:
            return f"Error: {error_msg}"

        full_prompt = (
            "You are a specialist in analyzing plots and visualizations of hybrid automata systems.\n\n"
            f"{question}"
        )

        img, img_bytes = self._resize_if_needed(img_bytes)

        if is_native_gemini(self.vision_model_id):
            return self._forward_gemini(img, full_prompt)
        else:
            return self._forward_litellm(img_bytes, full_prompt)


