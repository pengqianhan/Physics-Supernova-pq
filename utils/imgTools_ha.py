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
        "Given an image reference (placeholder like <image_10>) and a question, "
        "return the image expert's answer about that image."
        "When you need to measure quantities from an image, you MUST call this tool for Accurate Measurements: measuring youself alone is not accurate enough and could lead to errors!"
    )
    inputs = {
        "image_ref": {"type": "string", "description": "Placeholder identifying the image (e.g. <image_0>, <image_1>, ...)"},
        "question": {"type": "string", "description": "Question to ask about the image"},
    }
    output_type = "string"

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
        

    def _extract_image_bytes(self, image_ref: str) -> bytes | None:
        """Extract image bytes from markdown content using image reference like <image_1>.

        Args:
            image_ref: Image reference placeholder (e.g. <image_0>, <image_1>, ...)

        Returns:
            Image bytes from markdown content or None if not found
        """
        # image_ref ="<image_0>" ##TODO: remove this after testing
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

    def forward(self, image_ref: str, question: str) -> str:  # type: ignore[override]
        """Process image analysis request and return expert response."""
        img_bytes = self._extract_image_bytes(image_ref)
        # Resize image if it exceeds maximum resolution for better processing
        if self.max_short_side_pixels is not None:
            from PIL import Image
            from io import BytesIO

            if img_bytes is None:
                return f"Error: Could not find image {image_ref}; the parsed in image_ref should be like <image_N>."

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

        if img_bytes is None:
            return f"Error: Could not find image {image_ref}; the parsed in image_ref should be like <image_N>."

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