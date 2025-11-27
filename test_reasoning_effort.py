"""
Simple test to verify LiteLLMModel with reasoning_effort works in smolagents.
"""
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✓ dotenv loaded successfully")
except ImportError:
    print("dotenv not available, continuing without it")

from smolagents import LiteLLMModel, CodeAgent

def test_litellm_with_reasoning_effort():
    """Test LiteLLMModel with reasoning_effort parameter."""
    
    print("\n" + "="*50)
    print("Testing LiteLLMModel with reasoning_effort")
    print("="*50)
    
    # Create model without reasoning_effort, use the openai compatible format
    # model = LiteLLMModel(
    #     model_id="openai/gemini-flash-lite-latest",
    #     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    #     api_key=os.environ.get("GEMINI_API_KEY"),
    #     max_completion_tokens=24576,
    #     num_retries=3,
    #     timeout=1200,
    ##     reasoning_effort="low", # Only works with OpenAI o1/o3 models
    # )

    # Create model with reasoning_effort
    model = LiteLLMModel(
        model_id="gemini/gemini-flash-lite-latest",
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=8192,
        num_retries=3,
        timeout=1200,
        # thinking_budget = "1024" # gemini api format
        thinking_level = "low" # high, low
    )
    
    print(f"✓ Model created: {model.model_id}")
    
    # Test 1: Direct model call
    print("\n[Test 1] Direct model call...")
    messages = [
        {"role": "user", "content": "What is 2 + 2? Answer in one word."}
    ]
    
    try:
        response = model(messages)
        print(f"✓ Response received: {response.content[:100]}...")
        print(f"✓ Direct model call works!")
    except Exception as e:
        print(f"✗ Direct model call failed: {e}")
        return False
    
    # Test 2: With CodeAgent (simple task)
    print("\n[Test 2] CodeAgent with reasoning_effort model...")
    try:
        agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=3,
            verbosity_level=1
        )
        
        result = agent.run("Calculate the sum of 1, 2, and 3. Return just the number.")
        print(f"✓ Agent result: {result}")
        print(f"✓ CodeAgent works with reasoning_effort!")
    except Exception as e:
        print(f"✗ CodeAgent failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    return True


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("✗ Error: GEMINI_API_KEY not set in environment")
        exit(1)
    
    print(f"Using API key: {api_key[:10]}...")
    
    success = test_litellm_with_reasoning_effort()
    exit(0 if success else 1)

