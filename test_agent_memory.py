from smolagents import CodeAgent, LiteLLMModel
from smolagents.memory import (
    AgentMemory
)
from dotenv import load_dotenv
import os
from copy import deepcopy
load_dotenv()
  
model = LiteLLMModel(model_id="gemini/gemini-flash-lite-latest",api_key=os.environ.get("GEMINI_API_KEY"))
agent = CodeAgent(tools=[],model=model)
task = "compute the result of 3 + 5"
agent.run(task)

# read the memory
newest_input_messages = agent.write_memory_to_messages().copy()
# print("Newest Input Messages:")
# for msg in newest_input_messages:
#     print(f"{msg.role}: {msg.content}")

# read the task from the memory
print("Task from Memory:")
print(newest_input_messages[1]) # the second message is the task

# Function to summarize agent memory for completed problem parts
memory: AgentMemory = agent.memory

new_memory_step = deepcopy(memory.steps[0])
is_first_use = True
if is_first_use:
    new_memory_step.task += "You have finished a portion of the problem, and you have summarized your work on that part. Now you should continue to solve the rest parts of the problem. Please refer to previous output of 'finalize_part_answer' tool for previous context, in which the system and user have provided you with your previous work.\nNote that when calling reviewer, you should parse in your previous work to-be-reviewed through 'my_solution' input, or the reviewer would not be able to see it!"
# Clear memory and set up fresh start with context
agent.memory.reset()
agent.memory.steps = []
agent.memory.steps.append(new_memory_step)