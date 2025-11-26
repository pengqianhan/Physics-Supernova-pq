from smolagents import CodeAgent, InferenceClientModel
from smolagents import CodeAgent, LiteLLMModel
from PIL import Image
import os
import time
from smolagents.default_tools import Tool
from base64 import b64decode

# Import smolagents components for LLM model and message handling
from smolagents import (
    LiteLLMModel
)
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.markdown_utils import MarkdownMessage, load_trace_data_from_filepath
from utils.reviewTools_ha import ReviewRequestTool_ha
from dotenv import load_dotenv
import os
load_dotenv()
gemini_apikey = os.getenv('GEMINI_API_KEY')
# print(gemini_apikey)
model = LiteLLMModel(model_id="gemini/gemini-2.5-flash-lite", api_key=gemini_apikey) #

agent = CodeAgent(tools=[HybridAutomatonImageTool(), ReviewRequestTool_ha()], model=model, add_base_tools=True)

for toolName in agent.tools:
    agent.tools[toolName].worker_agent = agent

markdown_content = load_trace_data_from_filepath("utils/Dainarx_code/data_duffing")

# Set high res images in the agent, for the AskImageTool to use
# Dynamically add custom attribute to agent for data sharing with tools
# Docs: https://huggingface.co/docs/smolagents/en/tutorials/building_good_agents (agents support dynamic attributes)
print(markdown_content)
agent.markdown_content_high_res_image = markdown_content

# print(agent.tools)
# 移除特定的默认工具
tools_to_remove = ['web_search', 'visit_webpage']  # 例如移除这些工具
for tool_name in tools_to_remove:
    if tool_name in agent.tools:
        del agent.tools[tool_name]
# 加载本地图片
image = Image.open("sample_0.png")
task1 = """You are a hybrid automaton expert tasked with analyzing and improving hybrid automaton specifications.

System Configuration:
- System Name: {Duffing Oscillator}
- Number of Variables: {1}
- Number of Inputs: {1}

Instructions:
1. Carefully analyze the hybrid automaton structure shown in the provided image, you MUST use the hybrid_automaton_image_analysis_tool to analyze the image.
2. Identify potential improvements to the "mode"("id","eq"), "edge"("direction","condition","reset"),
3. Generate an improved version that maintains mathematical correctness and physical plausibility

Output Requirements:
- Return ONLY a valid Python dict representing the hybrid automaton specification
- Do NOT include any explanations, comments, or markdown formatting
- Ensure all mathematical expressions are syntactically correct"""
prompt1 = """
# Hybrid Automaton v0
PROMPT_HA_v0 = {
    "automaton": {  # automaton
        "var": "x1",  # variables list, separated by ','
        "input": "u1",  # input list, separated by ','
        "mode": [  # mode list
            {
                "id": 1,  # mode id
                "eq": "x1[1] = x1[0] + u1"
                # k-th order differential equation in the mode, separated by ','
                # cannot contain variables that are not defined in var, x[0] represents the original value of x, and x[k] is the k-th derivative
                # The left side of the equal sign is the highest order derivative 
                # The right side is the expression, does not support implicit functions
                # Any variable 'x' MUST be in the format x[k]
                # MUST provide ode for each variable
            }
        ],

        "edge": [
            {
                "direction": "1 -> 1",  # edge from mode u to mode v, represented as 'u -> v'
                "condition": "x1 >= 5",  # transition condition, cannot contain variables that are not defined in var
                "reset": {  # reset mapping for each variable, each variable has a list of reset values
                    "x1": [
                        "",
                        "x[1]"
                    ]
                }
            }
        ]
    },
    "config": {  # configuration parameters
        "dt": 0.001,  # discrete time step
        "total_time": 10.0,  # total sampling time 
        "dim": 1,  # dimension of ODE in "eq" field 
        "non_linear_items": ""  # additional nonlinear or cross terms in "eq" field
    }
}

# Hybrid Automaton v1

"""
task1 = task1 + prompt1
# agent.run(
#     task1,
#     additional_args={
#         "image": image
#     }
# )
agent.run(
    task1,
)
