import os
import time
import openai
from smolagents.default_tools import Tool

# Import smolagents components for LLM model and message handling
from smolagents.models import ChatMessage, MessageRole
from dotenv import load_dotenv
load_dotenv()

class HybridAutomatonImageTool(Tool):
    """Image expert to analyse the plot of the data collected from the system. You MUST call this tool when you need to measure quantities from an image."""

    name = "hybrid_automaton_image_analysis_tool"
    description = (
        "Given an image reference (placeholder like <image_10>) and a question, "
        "return the image expert's answer about that image."
        "When you need to measure quantities from an image, you MUST call this tool for Accurate Measurements: measuring youself alone is not accurate enough and could lead to errors!"
    )
    inputs = {
        "image_ref": {"type": "string", "description": "Placeholder identifying the image (e.g. <image_N>)"},
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
        

    def _extract_image_bytes(self, image_ref: str = None) -> bytes | None:
        """Read image bytes from sample_0.png file.

        Args:
            image_ref: Unused parameter kept for API compatibility

        Returns:
            Image bytes from sample_0.png or None if error occurs
        """
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(current_dir, "sample_0.png")

        # Read and return image bytes from sample_0.png
        try:
            with open(image_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: sample_0.png not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error reading sample_0.png: {str(e)}")
            return None

    def forward(self, image_ref: str, question: str) -> str:  # type: ignore[override]
        """Process image analysis request and return expert response."""
        img_bytes = self._extract_image_bytes(image_ref)
        # Resize image if it exceeds maximum resolution for better processing
        if self.max_short_side_pixels is not None:
            from PIL import Image
            from io import BytesIO

            if img_bytes is None:
                return "Error: Could not read sample_0.png image file."

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
            return "Error: Could not read sample_0.png image file."

        # Convert image bytes to base64 for OpenAI API
        import base64
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Prepare messages for vision model with system prompt and user query
        messages = [
            {
                "role": "system",
                "content": "You are an expert in dealing with image in hybrid automata analysis."
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