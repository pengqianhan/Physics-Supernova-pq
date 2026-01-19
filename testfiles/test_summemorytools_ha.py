"""Simple test for SummarizeMemoryTool"""
from utils.summemoryTools_ha import SummarizeMemoryTool
from smolagents.memory import AgentMemory, MemoryStep
from smolagents.models import ChatMessage, MessageRole
import json
from smolagents import CodeAgent, LiteLLMModel
import os
from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="gemini/gemini-flash-lite-latest",api_key=os.environ.get("GEMINI_API_KEY"))
agent = CodeAgent(tools=[],model=model)
task = "compute the result of 3 + 5"
agent.run(task)

# Test the SummarizeMemoryTool
print("=" * 60)
print("Testing SummarizeMemoryTool")
print("=" * 60)

# Create mock agent with simulated HA learning history
mock_agent = agent

print("\nMock agent memory contains:")
print(f"- Initial task: {agent.memory.steps[0].task[:80]}...")
print(f"- Number of refinement iterations: {len(agent.memory.steps) - 1}")

# Create summarization tool
summary_tool = SummarizeMemoryTool(worker_agent=agent)

print("\n" + "-" * 60)
print("Calling summarize tool (without memory reset)...")
print("-" * 60 + "\n")

# Test 1: Summarize without resetting memory
summary = summary_tool.forward(reset_memory=False)

print("\n" + "=" * 60)
print("Summary Output:")
print("=" * 60)
print(summary)

# Verify memory is still intact
print("\n" + "-" * 60)
print("Memory status after summarization (reset_memory=False):")
print(f"- Memory steps count: {len(agent.memory.steps)}")
print("-" * 60)

print("\n" + "=" * 60)
print("Testing with memory reset...")
print("=" * 60)

# Test 2: Summarize with memory reset
print("\n" + "-" * 60)
print("Calling summarize tool (with memory reset)...")
print("-" * 60 + "\n")

summary_with_reset = summary_tool.forward(reset_memory=True)

print("\n" + "=" * 60)
print("Summary Output (with reset):")
print("=" * 60)
print(summary_with_reset)

# Verify memory was reset
print("\n" + "-" * 60)
print("Memory status after summarization (reset_memory=True):")
print(f"- Memory steps count: {len(agent.memory.steps)}")
if len(agent.memory.steps) > 0:
    print(f"- First task preserved: {agent.memory.steps[0].task[:80]}...")
print("-" * 60)

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
