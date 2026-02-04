from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

# 读取本地图片文件
image_path = "data_all/non_linear/duffing/sample_0.png"

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
        "Analyze this hybrid automaton trajectory overlay plot. use hybrid automaton to describe it"
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    ),
)

parts_text = []

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print('part.text \n',part.text)
        parts_text.append(part.text)
        print('--------------------------------')
    if part.executable_code is not None:
        print('part.executable_code.code \n',part.executable_code.code)
        parts_text.append(part.executable_code.code)
        print('--------------------------------')
    if part.code_execution_result is not None:
        print('part.code_execution_result.output \n',part.code_execution_result.output)
        parts_text.append(part.code_execution_result.output)
        print('--------------------------------')

# save the response to a file, and read
save_parts_text = '\n'.join(parts_text)
with open('partsresponse.txt', 'w') as f:
    f.write(save_parts_text)
print('response saved to partsresponse.txt')

# save response to a file 
response_text = response.text
with open('response.txt', 'w') as f:
    f.write(response_text)
print('response saved to response.txt')
