import os
from litellm import completion
from dotenv import load_dotenv
load_dotenv()

response = completion(
    api_key=os.environ.get("GEMINI_API_KEY"),
    model="gemini/gemini-flash-lite-latest",## "gemini-flash-lite-latest" is not working
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

print(response.choices[0].message.content)

# Example 2: Image input with vision model
import base64

def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Using local image file (base64 encoded)
image_path = "data_all/ATVA/ball/sample_0.png"  # 修改为你的图片路径
base64_image = encode_image_to_base64(image_path)

response_with_image = completion(
    api_key=os.environ.get("GEMINI_API_KEY"),
    model="gemini/gemini-flash-lite-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the title of this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
print("Response (local image):", response_with_image.choices[0].message.content)