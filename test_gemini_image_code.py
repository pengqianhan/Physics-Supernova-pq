from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

# 读取本地图片文件
image_path = "evaluation_results/ATVA/ball/runs/20260202_152652_e357/best_iter_1/overlay_0.png"

# 方法1: 使用 PIL Image (推荐，更简洁)
image = Image.open(image_path)

# 方法2: 使用 types.Part.from_bytes (备选)
# with open(image_path, 'rb') as f:
#     image_bytes = f.read()
# image = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        image,
        "Analyze this hybrid automaton trajectory overlay plot. Describe what you see: the ground truth vs simulated trajectories, any mode transitions, and the quality of the fit."
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
