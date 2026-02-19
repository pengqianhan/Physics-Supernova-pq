from litellm import completion
import os
from dotenv import load_dotenv
load_dotenv()
## set ENV variables
response = completion(
  model="openai/kimi-k2.5",
  api_key=os.environ.get("MOONSHOT_API_KEY"),
  api_base="https://api.moonshot.cn/v1",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)

print(response.choices[0].message.content)