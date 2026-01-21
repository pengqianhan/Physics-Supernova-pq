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