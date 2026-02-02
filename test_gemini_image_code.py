from google import genai
from google.genai import types
import requests
from PIL import Image
import io
from dotenv import load_dotenv
import os
load_dotenv()

image_path = "https://goo.gle/instrument-img"
image_bytes = requests.get(image_path).content
image = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        image,
        "Zoom into the expression pedals and tell me how many pedals are there?"
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    ),
)

print('response \n',response.text)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print('part.text \n',part.text)
        print('--------------------------------')
    if part.executable_code is not None:
        print('part.executable_code.code \n',part.executable_code.code)
        print('--------------------------------')
    if part.code_execution_result is not None:
        print('part.code_execution_result.output \n',part.code_execution_result.output)
        print('--------------------------------')
