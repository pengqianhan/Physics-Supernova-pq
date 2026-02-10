import os
import time
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from smolagents.default_tools import Tool
from base64 import b64decode

# Import smolagents components for message handling
from .markdown_utils import MarkdownMessage
from dotenv import load_dotenv
load_dotenv()

class HybridAutomatonImageTool(Tool):
    """Image expert to analyse the plot of the data collected from the system. You MUST call this tool when you need to measure quantities from an image."""

    name = "hybrid_automaton_image_analysis"
    description = (
        "An advanced image expert tool equipped with a Python Code Execution Environment."
        "Given an image placeholder and a question, return the image expert's answer about that image. "
        "Accepts placeholders only: (1) <image_N> for original trace images from markdown, or "
        "(2) <iter_image_N> for evaluator plots registered during iteration. "
        "When you need to measure quantities or analyze trajectory comparisons, you MUST call this tool."
    )
    inputs = {
        "image_ref": {
            "type": "string",
            "description": "Image placeholder: <image_N> for trace images or <iter_image_N> for evaluator plots"
        },
        "question": {"type": "string", "description": "Question to ask about the image"},
    }
    output_type = "string"

    # Allowed image extensions for file path validation during registration
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    # Allowed directory prefixes for security (relative to repo root)
    ALLOWED_PATH_PREFIXES = ['evaluation_results/', 'data_all/']

    def __init__(self, worker_agent=None, vision_model_id: str = "gemini-3-flash-preview", max_short_side_pixels: int=9999):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to the main agent for accessing markdown content
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.vision_model_id = vision_model_id
        self.max_short_side_pixels = max_short_side_pixels  # Maximum image resolution for processing
        # Registry for iteration images (evaluator plots registered upstream)
        self._iteration_images: dict[int, bytes] = {}
        # Initialize genai.Client for Google Gemini API
        self.client = genai.Client(api_key=self.api_key)

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

    def _load_image_bytes_from_file(self, file_path: str) -> tuple[bytes | None, str]:
        """
        Read image bytes from a file path with security validation.
        Used internally during image registration.

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

    def register_iteration_image(self, index: int, source: str | bytes) -> tuple[bool, str]:
        """
        Register an evaluator plot image for later analysis via <iter_image_N> placeholder.

        Call this method upstream (before forward()) to make evaluator plots available
        for analysis using consistent placeholder-based references.

        Args:
            index: The iteration image index (used in <iter_image_N> placeholder)
            source: Either raw image bytes or a file path to load from

        Returns:
            Tuple of (success, error_message)

        Example:
            # Register from file path
            tool.register_iteration_image(0, "evaluation_results/run_1/overlay.png")
            # Then analyze using placeholder
            tool.forward("<iter_image_0>", "What does this plot show?")
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
        # Defensive runtime check (type checker marks as unreachable due to type annotation)
        return False, f"Invalid source type: {type(source)}. Expected bytes or str (file path)."  # type: ignore[unreachable]

    def clear_iteration_images(self) -> None:
        """
        Clear all registered iteration images.

        Call this between evaluation runs to reset the image registry.
        """
        self._iteration_images.clear()

    def get_registered_iteration_indices(self) -> list[int]:
        """
        Get list of registered iteration image indices.

        Returns:
            List of indices that have registered images
        """
        return sorted(self._iteration_images.keys())

    def _extract_image_bytes_from_placeholder(self, image_ref: str) -> tuple[bytes | None, str]:
        """
        Extract image bytes using placeholder reference.

        Supports placeholder types:
        - <image_N>: Original trace images from markdown content
        - <iter_image_N>: Evaluator plots registered via register_iteration_image()
        - <iter_image_X_Y>: Compound format where index = X * 100 + Y (for multi-file evaluation)

        Args:
            image_ref: Image reference placeholder

        Returns:
            Tuple of (image_bytes or None, error_message)
        """
        # Handle <iter_image_N> or <iter_image_X_Y> placeholders (evaluator plots)
        if image_ref.startswith("<iter_image_") and image_ref.endswith(">"):
            inner = image_ref[len("<iter_image_"):-1]
            try:
                if '_' in inner:
                    # Format: <iter_image_X_Y> -> index = X * 100 + Y
                    parts = inner.split('_')
                    idx = int(parts[0]) * 100 + int(parts[1])
                else:
                    # Format: <iter_image_N> (backward compatible)
                    idx = int(inner)
            except (ValueError, IndexError):
                return None, f"Invalid iteration image placeholder format: {image_ref}"

            if idx not in self._iteration_images:
                available = self.get_registered_iteration_indices()
                if available:
                    return None, f"Iteration image {idx} not registered. Available: {available}"
                else:
                    return None, f"No iteration images registered. Call register_iteration_image() first."
            return self._iteration_images[idx], ""

        # Handle <image_N> placeholders (original markdown images)
        if image_ref.startswith("<image_") and image_ref.endswith(">"):
            try:
                idx = int(image_ref[len("<image_"):-1])
            except ValueError:
                return None, f"Invalid image placeholder format: {image_ref}"
        elif image_ref.isdigit():
            idx = int(image_ref)  # Handle plain number references
        else:
            return None, f"Invalid placeholder format: {image_ref}. Use <image_N> or <iter_image_N>."

        # Extract from markdown content
        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            return None, "No markdown content available. Ensure worker_agent is set with markdown_content_high_res_image."

        md: MarkdownMessage = self.worker_agent.markdown_content_high_res_image
        img_blocks = [it for it in md.content if it.get("type") == "image_url"]

        if not (0 <= idx < len(img_blocks)):
            return None, f"Image index {idx} out of range. Available: 0-{len(img_blocks)-1}"

        data_url = img_blocks[idx]["image_url"]["url"]
        if data_url.startswith("data:image"):
            base64_part = data_url.split(",", 1)[1]
            return b64decode(base64_part), ""

        return None, f"Image {idx} has invalid data URL format."

    def _extract_image_bytes(self, image_ref: str) -> tuple[bytes | None, str]:
        """
        Extract image bytes from a placeholder reference.

        Only accepts placeholder-based inputs:
        - <image_N>: Original trace images from markdown content
        - <iter_image_N>: Evaluator plots registered via register_iteration_image()

        File paths are NOT supported in forward(). To analyze evaluator plots,
        register them first using register_iteration_image(), then reference
        them via <iter_image_N> placeholders.

        Args:
            image_ref: Image placeholder reference

        Returns:
            Tuple of (image_bytes or None, error_message)
        """
        return self._extract_image_bytes_from_placeholder(image_ref)

    def forward(self, image_ref: str, question: str) -> str:  # type: ignore[override]
        """Process image analysis request and return expert response."""
        # Extract image bytes from placeholder (<image_N> or <iter_image_N>)
        img_bytes, error_msg = self._extract_image_bytes(image_ref)

        if img_bytes is None:
            return f"Error: {error_msg}"

        # Convert bytes to PIL.Image for genai.Client
        img = Image.open(BytesIO(img_bytes))

        # Resize image if it exceeds maximum resolution for better processing
        if self.max_short_side_pixels is not None:
            width, height = img.size
            if min(width, height) > self.max_short_side_pixels:
                scale = self.max_short_side_pixels / min(width, height)
                new_size = (int(width * scale), int(height * scale))
                # Use BILINEAR resampling (ANTIALIAS deprecated in newer PIL versions)
                img = img.resize(new_size, Image.Resampling.BILINEAR)

        # Prepare prompt with system context
        full_prompt = (
            "You are a specialist in analyzing plots and visualizations of hybrid automata systems.\n\n"
            f"{question}"
        )

        # Retry logic for robust image analysis
        max_try = 3
        for _ in range(max_try):
            # Generate response from vision model using genai.Client
            try:
                response = self.client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[img, full_prompt],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)]),
                )
                parts_text = []
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        parts_text.append(part.text)
                    if part.executable_code is not None:
                        parts_text.append(part.executable_code.code)
                    if part.code_execution_result is not None:
                        parts_text.append(part.code_execution_result.output)
                all_parts_text = '\n'.join(parts_text)
                # Extract response text
                if all_parts_text and all_parts_text.strip():
                    return all_parts_text.strip()
            except Exception as e:
                print(f"Error during vision model generation: {str(e)}")
                time.sleep(5)  # Wait before retry
        return "No response"

if __name__ == "__main__":
    tool = HybridAutomatonImageTool()

    # Example: Register an evaluator plot, then analyze via placeholder
    # tool.register_iteration_image(0, "evaluation_results/run_1/overlay.png")
    # print(tool.forward("<iter_image_0>", "What does this plot show?"))

    # For testing without actual image, this will show an error message
    print(tool.forward("<iter_image_0>", "What is the plot of the data collected from the system?"))