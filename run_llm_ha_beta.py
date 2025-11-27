import os
import prompts_ha
# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Load from .env file in the current directory
    print("dotenv loaded successfully from .env file")
    print(f"Currently using api key: {os.environ.get('GEMINI_API_KEY', 'Not Set')[:20]}...")
except ImportError:
    # dotenv not available, continue without it
    print("dotenv not available, continuing without it")
    pass
except Exception:
    # .env file doesn't exist or other error, continue without raising exception
    print("Error loading .env file, continuing without it")
    pass

import argparse

# for type hints
from typing import List
from smolagents.default_tools import Tool
from smolagents import (
    MultiStepAgent,
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)

# deal with markdown contents
from utils import MarkdownMessage
from utils.markdown_utils import load_trace_data_from_filepath

# Import HA-specific tools
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.reviewTools_ha import ReviewRequestTool_ha


# Tool mapping for HA-specific tools
TOOLNAME2TOOL = {
    'hybrid_automaton_image_analysis': HybridAutomatonImageTool,
    'ask_review_expert_ha': ReviewRequestTool_ha,
}


def _create_HA_agent(Tools_list: List[type[Tool]],
                     markdown_content: MarkdownMessage,
                     model_id: str = "gemini/gemini-flash-lite-latest",
                     managed_agents_list: List[MultiStepAgent] = None,
                     max_steps: int = 80,
                     **kwargs) -> ToolCallingAgent | CodeAgent:

    # LLM to use for the agent
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200,
        thinking_level = "low" # high, low
    )

    # tools for the manager agent
    tools = []
    print('Tools_list: ', Tools_list)
    print('kwargs: ', kwargs)
    for tool in Tools_list:
        if tool.__name__ == "HybridAutomatonImageTool" and "image_tool_model" in kwargs:
            tools.append(tool(vision_model_id=kwargs["image_tool_model"]))
        elif tool.__name__ == "ReviewRequestTool_ha" and "review_tool_model" in kwargs:
            tools.append(tool(review_model_id=kwargs["review_tool_model"]))
        else:
            tools.append(tool())

    # initialize the manager agent
    manager_agent_kwargs = dict(
        model=model,
        tools=tools,
        max_steps=max_steps,
        verbosity_level=2,
        name="ha_learning_agent",
        description="",
        managed_agents=managed_agents_list,
        add_base_tools=True  # Add base tools like python_interpreter
    )

    if kwargs["manager_type"] == "CodeAgent":
        manager_agent_kwargs["additional_authorized_imports"] = [
            "os", "sys", "time", "argparse", "pathlib",
            "matplotlib.pyplot", "numpy", "pandas", "json","scipy"
        ]
        managerAgent = CodeAgent(**manager_agent_kwargs)
    elif kwargs["manager_type"] == "ToolCallingAgent":
        manager_agent_kwargs["max_tool_threads"] = 1
        managerAgent = ToolCallingAgent(**manager_agent_kwargs)
    else:
        raise ValueError(f"Unknown manager type: {kwargs['manager_type']}. Must be 'ToolCallingAgent' or 'CodeAgent'.")

    # Inject agent reference into each tool (delayed injection pattern)
    # This allows tools like ReviewRequestTool_ha to access agent.markdown_content_high_res_image
    for toolName in managerAgent.tools:
        managerAgent.tools[toolName].worker_agent = managerAgent

    # Set high res images in the agent, for the HybridAutomatonImageTool to use
    # Dynamically add custom attribute to agent for data sharing with tools
    # Docs: https://huggingface.co/docs/smolagents/en/tutorials/building_good_agents (agents support dynamic attributes)
    managerAgent.markdown_content_high_res_image = markdown_content

    # Remove unwanted default tools
    tools_to_remove = kwargs.get("tools_to_remove", ['web_search', 'visit_webpage'])
    for tool_name in tools_to_remove:
        if tool_name in managerAgent.tools:
            del managerAgent.tools[tool_name]

    return managerAgent


def get_managed_agents_list(managed_agents_list: List[str] = None,
                            managed_agents_list_model_id: str = None) -> List[MultiStepAgent]:
    if managed_agents_list is None:
        return []

    managed_agents = []
    for agent_name in managed_agents_list:
        # LLM model
        model = LiteLLMModel(
            model_id=managed_agents_list_model_id,
            api_key=os.environ.get("GEMINI_API_KEY"),
            max_completion_tokens=24576,
            num_retries=3,
            timeout=1200
        )
        # managed agent
        managed_agent = CodeAgent(
            tools=[],
            model=model,
            name=agent_name,
            additional_authorized_imports=["os", "sys", "time", "argparse", "pathlib",
                                          "matplotlib.pyplot", "numpy", "pandas", "json","scipy"],
            description=f"I am a managed agent with name {agent_name}. I can assist with code-related tasks",
            max_steps=80,
            verbosity_level=2,
        )
        managed_agents.append(managed_agent)

    return managed_agents


# create the agent
def create_agent(model_id: str = "gemini/gemini-flash-lite-latest",
                input_data_path: str = None,
                tools_list: List[str] = [],
                managed_agents_list: List[str] = None,
                managed_agents_list_model_id: str = None,
                **kwargs) -> ToolCallingAgent | CodeAgent:
    '''
    Create a Hybrid Automaton learning agent.

    Args:
        model_id: Model ID for the main agent
        input_data_path: Path to the trace data directory
        tools_list: List of tool names to use
        managed_agents_list: List of managed agent names
        managed_agents_list_model_id: Model ID for managed agents
        **kwargs: Additional arguments (manager_type, image_tool_model, review_tool_model, etc.)

    Returns:
        Configured ToolCallingAgent or CodeAgent
    '''
    # Load trace data with high res images
    markdown_content = load_trace_data_from_filepath(input_data_path)

    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]
    haAgent = _create_HA_agent(
        Tools_list=ToolsList,
        markdown_content=markdown_content,
        model_id=model_id,
        managed_agents_list=get_managed_agents_list(managed_agents_list, managed_agents_list_model_id),
        **kwargs
    )
    return haAgent


# obtain task string for the agent to run
def obtain_task(system_name: str = "Duffing Oscillator",
               num_variables: int = 1,
               num_inputs: int = 1,
               initial_ha_spec: str = None,
               tools_list: List[str] = [],
               managed_agents_list: List[str] = None,
               manager_type: str = "CodeAgent") -> str:
    '''
    Generate task prompt for HA learning agent.

    Args:
        system_name: Name of the dynamical system
        num_variables: Number of state variables
        num_inputs: Number of input variables
        initial_ha_spec: Initial HA specification dictionary (optional)
        tools_list: List of tool names
        managed_agents_list: List of managed agent names
        manager_type: Type of manager agent

    Returns:
        Task prompt string
    '''
    print("tools_list: ", tools_list)

    # Base task prompt
    task = f"""You are a hybrid automaton expert tasked with analyzing and improving hybrid automaton specifications.

Instructions:
1. Carefully analyze the hybrid automaton structure shown in the provided image"""

    # Add tool-specific prompts
    HA_IMAGE_TOOL_PROMPT = ", you MUST use the hybrid_automaton_image_analysis tool to analyze the image."
    REVIEW_TOOL_PROMPT = " When you need expert review of your HA specification, you MUST call the `ask_review_expert_ha` tool."

    if HybridAutomatonImageTool in [TOOLNAME2TOOL[x] for x in tools_list]:
        task += HA_IMAGE_TOOL_PROMPT

    task += """
2. Identify potential improvements to the "mode"("id","eq"), "edge"("direction","condition","reset")
3. Generate an improved version that maintains mathematical correctness and physical plausibility"""

    if ReviewRequestTool_ha in [TOOLNAME2TOOL[x] for x in tools_list]:
        task += REVIEW_TOOL_PROMPT

    # Add output requirements
    task += """

Output Requirements:
- Return ONLY a valid Python dict representing the hybrid automaton specification
- Do NOT include any explanations, comments, or markdown formatting
- Ensure all mathematical expressions are syntactically correct"""

    # Add managed agents prompt
    if managed_agents_list and len(managed_agents_list) > 0:
        MANAGE_AGENT_PROMPT = f"\n\nYou may use the managed Code Agent: {managed_agents_list} to assist you with code-related tasks."
        task += MANAGE_AGENT_PROMPT

    # Add self code agent prompt
    if manager_type == "CodeAgent":
        SELF_IS_CODE_AGENT_PROMPT = "\n\nYou can use Python Code to execute programs, which may help with your task-solving process."
        task += SELF_IS_CODE_AGENT_PROMPT

    # Add initial HA specification if provided
    if initial_ha_spec:
        task += f"""

# Initial Hybrid Automaton Specification (v0)
PROMPT_HA_v0 = {initial_ha_spec}

# Task
Generate an improved version (v1) based on the analysis."""
    else:
        # Add default template
        task += """

# Hybrid Automaton Template
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
                        "x1[0]"
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

# Task
Generate an improved version (v1) based on the analysis."""
    system_config_prompt = f"""\n\n
System Configuration:
- System Name: {{{system_name}}}
- Number of Variables: {{{num_variables}}}
- Number of Inputs: {{{num_inputs}}}
\n\n
"""
    task =  task + system_config_prompt
    return task


def parse_args():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "utils", "Dainarx_code", "data_duffing")

    ap = argparse.ArgumentParser(description="Run the HA Learning Agent with specified tools and model.")
    ap.add_argument(
        "--input-data-path",
        type=str,
        default=default_data_path,
        help="Path to the trace data directory.",
    )
    ap.add_argument(
        "--manager-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID to use for the agent.",
    )

    # Choose between ToolCallingAgent and CodeAgent for the manager agent
    ap.add_argument(
        "--manager-type",
        type=str,
        default="CodeAgent",
        choices=["ToolCallingAgent", "CodeAgent"],
        help="Type of agent to create (default: CodeAgent).",
    )

    # Tools available for the manager agent
    ap.add_argument(
        "--tools-list",
        type=str,
        nargs='*',
        default=["hybrid_automaton_image_analysis", "ask_review_expert_ha"],
        help="List of tool names to use in the agent.",
    )

    # LLM model ids for the tools
    ap.add_argument(
        "--image-tool-model",
        type=str,
        default="gemini-flash-lite-latest",
        help="Model ID to use for the image analysis tool (Gemini API format).",
    )
    ap.add_argument(
        "--review-tool-model",
        type=str,
        default="gemini-flash-lite-latest",
        help="Model ID to use for the review tool (Gemini API format).",
    )

    # agent names of managed agents
    ap.add_argument(
        "--managed-agents-list",
        type=str,
        nargs='*',
        default=[],
        help="List of managed agents to use in the agent.",
    )

    # LLM model ids for the managed agents
    ap.add_argument(
        "--managed-agents-list-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID to use for managed agents.",
    )

    # System configuration parameters
    ap.add_argument(
        "--system-name",
        type=str,
        default="Duffing Oscillator",
        help="Name of the dynamical system.",
    )
    ap.add_argument(
        "--num-variables",
        type=int,
        default=1,
        help="Number of state variables.",
    )
    ap.add_argument(
        "--num-inputs",
        type=int,
        default=1,
        help="Number of input variables.",
    )

    # HA specification file (optional)
    ap.add_argument(
        "--initial-ha-spec-file",
        type=str,
        default=None,
        help="Path to JSON file containing initial HA specification (optional).",
    )

    # Tools to remove
    ap.add_argument(
        "--tools-to-remove",
        type=str,
        nargs='*',
        default=['web_search', 'visit_webpage'],
        help="List of default tools to remove from the agent.",
    )

    args = ap.parse_args()

    if not args.input_data_path:
        raise ValueError("You must provide a data path with --input-data-path.")
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


def main():
    args = parse_args()
    print(f"Running HA Learning Agent with model: {args.manager_model}, tools: {args.tools_list}, data path: {args.input_data_path}")

    # Load initial HA spec if provided
    initial_ha_spec = prompts_ha.initial_ha_spec_prompt

    # Create the agent
    managerAgent = create_agent(
        model_id=args.manager_model,
        input_data_path=args.input_data_path,
        tools_list=args.tools_list,
        image_tool_model=args.image_tool_model,
        review_tool_model=args.review_tool_model,
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
        managed_agents_list_model_id=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else None,
        manager_type=args.manager_type,
        tools_to_remove=args.tools_to_remove,
    )

    # Obtain task
    task = obtain_task(
        system_name=args.system_name,
        num_variables=args.num_variables,
        num_inputs=args.num_inputs,
        initial_ha_spec=initial_ha_spec,
        tools_list=args.tools_list,
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
        manager_type=args.manager_type,
    )
    # save the task to a file
    with open("task.txt", "w") as f:
        f.write(task)
    print(f"Saved task to task.txt")

    # Run the agent
    managerAgent.run(task)


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_llm_ha_beta.py --input-data-path utils/Dainarx_code/data_duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis ask_review_expert_ha
