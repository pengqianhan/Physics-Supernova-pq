import os
from smolagents import CodeAgent, LiteLLMModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

gemini_apikey = os.environ.get('GEMINI_API_KEY')
e2b_apikey = os.environ.get('E2B_API_KEY')
print(f"Gemini API key: {gemini_apikey}")
print(f"E2B API key: {e2b_apikey}")

model = LiteLLMModel(model_id="gemini/gemini-2.5-flash-lite", api_key=gemini_apikey)

with CodeAgent(model=model, tools=[], executor_type="e2b") as agent:
    agent.run("Can you give me the 100th Fibonacci number?")