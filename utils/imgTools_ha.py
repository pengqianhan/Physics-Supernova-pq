import os
import time
import openai
from smolagents.default_tools import Tool
from base64 import b64decode

# Import smolagents components for LLM model and message handling
from smolagents.models import ChatMessage, MessageRole
from .markdown_utils import MarkdownMessage
from dotenv import load_dotenv
load_dotenv()

class HybridAutomatonImageTool(Tool):
    """Image expert to analyse the plot of the data collected from the system. You MUST call this tool when you need to measure quantities from an image."""

    name = "hybrid_automaton_image_analysis"
    description = (
        "Given an image reference and a question, return the image expert's answer about that image. "
        "Supports two input types: (1) placeholder like <image_N> for trace images, or "
        "(2) file path like 'evaluation_results/runs/.../overlay.png' for evaluator plots. "
        "When you need to measure quantities or analyze trajectory comparisons, you MUST call this tool."
    )
    inputs = {
        "image_ref": {
            "type": "string",
            "description": "Image reference: either <image_N> placeholder or file path to evaluation plot"
        },
        "question": {"type": "string", "description": "Question to ask about the image"},
    }
    output_type = "string"

    # Allowed image extensions for file path mode
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    # Allowed directory prefixes for security (relative to repo root)
    ALLOWED_PATH_PREFIXES = ['evaluation_results/', 'data_all/']

    def __init__(self, worker_agent=None, vision_model_id: str = "models/gemini-flash-lite-latest", max_short_side_pixels: int=9999):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to the main agent for accessing markdown content
        # Initialize OpenAI client for Gemini API
        self.client = openai.OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY") if os.getenv("GEMINI_API_KEY") else "<<<<<<your_api_key>>>>>>>",
        )
        self.vision_model_id = vision_model_id
        self.max_short_side_pixels = max_short_side_pixels  # Maximum image resolution for processing

    def _is_valid_file_path(self, path: str) -> tuple[bool, str]:
        """
        Validate that a file path is allowed and secure.

        Args:
            path: The path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Reject path traversal attempts
        if '..' in path:
            return False, "Path traversal (..) is not allowed for security reasons."

        # Check file extension
        _, ext = os.path.splitext(path.lower())
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File extension '{ext}' not allowed. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"

        # Check if path starts with allowed prefix
        path_normalized = path.replace('\\', '/')
        if not any(path_normalized.startswith(prefix) for prefix in self.ALLOWED_PATH_PREFIXES):
            # Also allow absolute paths within allowed directories
            abs_path = os.path.abspath(path)
            cwd = os.getcwd()
            for prefix in self.ALLOWED_PATH_PREFIXES:
                allowed_abs = os.path.abspath(os.path.join(cwd, prefix))
                if abs_path.startswith(allowed_abs):
                    return True, ""
            return False, f"Path must start with one of: {', '.join(self.ALLOWED_PATH_PREFIXES)}"

        return True, ""

    def _extract_image_bytes_from_file(self, file_path: str) -> tuple[bytes | None, str]:
        """
        Read image bytes from a file path with security validation.

        Args:
            file_path: Path to the image file

        Returns:
            Tuple of (image_bytes or None, error_message)
        """
        # Validate path security
        is_valid, error_msg = self._is_valid_file_path(file_path)
        if not is_valid:
            return None, error_msg

        # Resolve to absolute path
        abs_path = os.path.abspath(file_path)

        # Check file exists
        if not os.path.isfile(abs_path):
            return None, f"File not found: {file_path}"

        # Read file
        try:
            with open(abs_path, 'rb') as f:
                return f.read(), ""
        except Exception as e:
            return None, f"Failed to read file: {str(e)}"

    def _extract_image_bytes_from_placeholder(self, image_ref: str) -> bytes | None:
        """
        Extract image bytes from markdown content using image reference like <image_1>.

        Args:
            image_ref: Image reference placeholder (e.g. <image_N> or plain number)

        Returns:
            Image bytes from markdown content or None if not found
        """
        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            return None
        md: MarkdownMessage = self.worker_agent.markdown_content_high_res_image
        # Parse image reference to extract index (supports both <image_N> and plain numbers)
        idx = None
        if image_ref.startswith("<image_") and image_ref.endswith(">"):
            try:
                idx = int(image_ref.strip("<image_>"))  # Convert to 0-based index
            except ValueError:
                pass
        else:
            try:
                idx = int(image_ref)  # Handle plain number references
            except ValueError:
                pass
        if idx is None:
            return None
        # Extract all image blocks from markdown content
        img_blocks = [it for it in md.content if it.get("type") == "image_url"]
        if 0 <= idx < len(img_blocks):
            data_url = img_blocks[idx]["image_url"]["url"]
            if data_url.startswith("data:image"):
                base64_part = data_url.split(",", 1)[1]  # Remove data URL prefix
                return b64decode(base64_part)  # Decode base64 to bytes
        return None

    def _extract_image_bytes(self, image_ref: str) -> tuple[bytes | None, str]:
        """
        Extract image bytes from either a placeholder or file path.

        Args:
            image_ref: Image reference - either <image_N> placeholder or file path

        Returns:
            Tuple of (image_bytes or None, error_message)
        """
        # Determine if this is a placeholder or file path
        is_placeholder = (
            image_ref.startswith("<image_") and image_ref.endswith(">")
        ) or image_ref.isdigit()

        if is_placeholder:
            # Handle placeholder reference
            img_bytes = self._extract_image_bytes_from_placeholder(image_ref)
            if img_bytes is None:
                return None, f"Could not find image {image_ref}. Valid format: <image_N> where N is 0-indexed."
            return img_bytes, ""
        else:
            # Handle file path reference
            return self._extract_image_bytes_from_file(image_ref)

    def forward(self, image_ref: str, question: str) -> str:  # type: ignore[override]
        """Process image analysis request and return expert response."""
        # Extract image bytes (supports both <image_N> placeholders and file paths)
        img_bytes, error_msg = self._extract_image_bytes(image_ref)

        if img_bytes is None:
            return f"Error: {error_msg}"

        # Resize image if it exceeds maximum resolution for better processing
        if self.max_short_side_pixels is not None:
            from PIL import Image
            from io import BytesIO

            # Check image size and resize if necessary
            img = Image.open(BytesIO(img_bytes))
            width, height = img.size
            if min(width, height) > self.max_short_side_pixels:
                scale = self.max_short_side_pixels / min(width, height)
                new_size = (int(width * scale), int(height * scale))
                # Use BILINEAR resampling (ANTIALIAS deprecated in newer PIL versions)
                img = img.resize(new_size, Image.Resampling.BILINEAR)

                # Convert resized image back to bytes
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_bytes = buffer.getvalue()

        # Convert image bytes to base64 for OpenAI API
        import base64
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare messages for vision model with system prompt and user query
        messages = [
            {
                "role": "system",
                "content": "You are a specialist in analyzing plots and visualizations of hybrid automata systems."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        # Retry logic for robust image analysis
        max_try = 3
        for _ in range(max_try):
            # Generate response from vision model
            try:
                response = self.client.chat.completions.create(
                    model=self.vision_model_id,
                    messages=messages,
                    max_tokens=8192,
                )
                if response.choices[0].message.content and response.choices[0].message.content.strip():
                    return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error during vision model generation: {str(e)}")
                time.sleep(5)  # Wait before retry
        return "No response"

if __name__ == "__main__":
    tool = HybridAutomatonImageTool()
    print(tool.forward("sample_0.png", "What is the plot of the data collected from the system?"))