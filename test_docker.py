from smolagents import InferenceClientModel, CodeAgent
from smolagents import LiteLLMModel
import os
from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(
    model_id="gemini/gemini-flash-lite-latest",
    api_key=os.environ.get("GEMINI_API_KEY"),
    max_completion_tokens=4096,
)

with CodeAgent(model=model, tools=[], executor_type="docker") as agent:
    agent.run("Can you give me the 100th Fibonacci number?")