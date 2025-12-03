import os
import sys
import json
import prompts_ha
from prompts_ha.prompts import HA_SPEC_DOCUMENTATION, initial_ha_spec_prompt
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
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress

# Import HA-specific tools
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.reviewTools_ha import ReviewRequestTool_ha
from utils.summemoryTools_ha import SummarizeMemoryTool


# Tool mapping for HA-specific tools
TOOLNAME2TOOL = {
    'hybrid_automaton_image_analysis': HybridAutomatonImageTool,
    'ask_review_expert_ha': ReviewRequestTool_ha,
    'summarize_ha_iterations': SummarizeMemoryTool,
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
        elif tool.__name__ == "SummarizeMemoryTool" and "summarize_tool_model" in kwargs:
            tools.append(tool(summarize_model_id=kwargs["summarize_tool_model"]))
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
        add_base_tools=False  # False means only use the 
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

    # Remove unwanted default tools or make add_base_tools=False in manager_agent_kwargs
    # tools_to_remove = kwargs.get("tools_to_remove", ['web_search', 'visit_webpage'])
    # for tool_name in tools_to_remove:
    #     if tool_name in managerAgent.tools:
    #         del managerAgent.tools[tool_name]

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


# obtain task string and images for the agent to run
def obtain_task_and_images(input_data_path: str = None,
                           system_name: str = "Duffing Oscillator",
                           num_variables: int = 1,
                           num_inputs: int = 1,
                           initial_ha_spec: str = None,
                           tools_list: List[str] = [],
                           managed_agents_list: List[str] = None,
                           manager_type: str = "CodeAgent") -> tuple[str, list]:
    '''
    Generate task prompt and compressed images for HA learning agent.

    Args:
        input_data_path: Path to the trace data directory
        system_name: Name of the dynamical system
        num_variables: Number of state variables
        num_inputs: Number of input variables
        initial_ha_spec: Initial HA specification dictionary (optional)
        tools_list: List of tool names
        managed_agents_list: List of managed agent names
        manager_type: Type of manager agent

    Returns:
        Tuple of (task prompt string, list of compressed images)
    '''
    print("tools_list: ", tools_list)

    # Load trace data with high res images
    markdown_content = load_trace_data_from_filepath(input_data_path)

    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]

    # Get task and images (to parse into agents) from the markdown content
    trace_data_text = markdown_to_plaintext(markdown_content)
    compressed_trace_images = markdown_images_compress(markdown_content, max_short_side_pixels=1080)

    # Base task prompt - Professional system identification framing
    task = f"""# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
You are given time-series trajectory data from an unknown hybrid dynamical system. Your task is to:
1. **Identify discrete modes** (operating regimes with distinct continuous dynamics)
2. **Infer mode-specific ODEs** (differential equations governing each regime)
3. **Determine switching conditions** (guard predicates triggering mode transitions)
4. **Specify reset maps** (state updates upon mode transitions)

## Available Data
The following trace data visualizations are provided (reference images using placeholders: `<image_0>`, `<image_1>`, etc.):
- State variable trajectories over time
- Input signals (if applicable)
- Potential mode-switch indicators (discontinuities, slope changes)

## Analysis Workflow"""

    # Add tool-specific prompts
    HA_IMAGE_TOOL_PROMPT = ", you MUST use the hybrid_automaton_image_analysis tool to analyze the image."
    REVIEW_TOOL_PROMPT = " When you need expert review of your hybrid automaton specification, you MUST call the `ask_review_expert_ha` tool."
    if ReviewRequestTool_ha in ToolsList:
        REVIEW_TOOL_PROMPT += """
**MANDATORY BEFORE FINALIZATION**: You MUST call `ask_review_expert_ha` at least once before submitting your final answer to ensure specification correctness and completeness."""

    task += HA_IMAGE_TOOL_PROMPT if HybridAutomatonImageTool in ToolsList else ""
    task += REVIEW_TOOL_PROMPT if ReviewRequestTool_ha in ToolsList else ""

    task += """

## HA Refinement Guidelines
Focus your analysis on:
1. **Mode Dynamics (`mode.eq`)**: Derive ODEs that match observed trajectory slopes and curvatures
2. **Guard Conditions (`edge.condition`)**: Identify state thresholds where switching occurs
3. **Transition Structure (`edge.direction`)**: Determine mode connectivity (self-loops, bidirectional, unidirectional)
4. **Reset Maps (`edge.reset`)**: Specify whether states are continuous or discontinuous across transitions

## Quality Criteria
Your HA specification will be evaluated on:
- **Trajectory Matching**: Simulated output should closely follow ground truth data
- **Mode Detection Accuracy**: Correct identification of switching instants (`tc` metric)
- **State Error Minimization**: Low `mean_diff` and `max_diff` between predicted and actual states"""

    # Add output requirements with clear format specification
    task += """

## Output Format Requirements
Return a **valid Python dictionary** with the following structure:
```python
{
    "automaton": {
        "var": "x1, x2, ...",      # State variables
        "input": "u1, u2, ...",    # Input signals
        "mode": [{"id": 1, "eq": "..."}],  # Mode dynamics
        "edge": [{"direction": "1 -> 2", "condition": "...", "reset": {...}}]  # Transitions
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "dim": 1,
        "need_reset": true,
        "non_linear_items": "..."
    }
}
```
**CRITICAL**: Return ONLY the Python dict. No markdown formatting, no explanations, no code blocks in the final answer."""

    # Add managed agents prompt
    if managed_agents_list and len(managed_agents_list) > 0:
        MANAGE_AGENT_PROMPT = f"""

## Computational Resources
You have access to managed Code Agent(s): `{managed_agents_list}`
Use them for numerical computations, curve fitting, or complex mathematical derivations."""
        task += MANAGE_AGENT_PROMPT

    # Add self code agent prompt
    if manager_type == "CodeAgent":
        SELF_IS_CODE_AGENT_PROMPT = "\n\n## Code Execution Capability\nYou can use Python Code to execute programs, which may help with your task-solving process."
        task += SELF_IS_CODE_AGENT_PROMPT

    # Add HA specification format documentation and initial spec
    task += f"""

## HA Specification Format Reference
{HA_SPEC_DOCUMENTATION}

## Initial HA Specification (v0 - Baseline)
The following is an initial template/guess. Analyze the trace data and refine this into an accurate model:

```python
{initial_ha_spec_prompt}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data. Apply systematic refinement: analyze discrepancies → hypothesize corrections → validate improvements."""

    system_config_prompt = f"""

## Target System Configuration
| Parameter | Value |
|-----------|-------|
| System Name | {system_name} |
| State Variables | {num_variables} |
| Input Signals | {num_inputs} |
"""
    task = task + system_config_prompt

    # Add trace data description
    task += f"""

## Observed Trace Data
{trace_data_text}
"""

    return task, compressed_trace_images


def evaluate_ha_specification(agent_result, input_data_path: str, output_dir: str = None) -> bool:
    """
    Evaluate the generated Hybrid Automaton specification against ground truth data.

    Args:
        agent_result: Result from the agent.run() call (can be dict or str)
        input_data_path: Path to the directory containing test .npz files
        output_dir: Directory to save evaluation results (default: './evaluation_results')

    Returns:
        bool: True if evaluation succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Testing the generated Hybrid Automaton specification")
    print("=" * 80)

    # Import HA evaluation module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'Dainarx_code'))
    from HA_evaluation import HAEvaluator

    # Extract HA specification from agent result
    ha_specification = None

    # Try to extract HA spec from result (smolagents returns different formats)
    if isinstance(agent_result, dict):
        ha_specification = agent_result
    elif isinstance(agent_result, str):
        # Try to parse as JSON
        try:
            ha_specification = json.loads(agent_result)
        except:
            print(f"Warning: Could not parse agent result as JSON. Result type: {type(agent_result)}")
            print(f"Result content: {agent_result}")
    else:
        print(f"Warning: Unexpected result type: {type(agent_result)}")
        print(f"Result content: {agent_result}")

    # Validate HA specification structure
    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        print("\nWarning: Could not extract valid HA specification from agent output for evaluation")
        print("Expected dictionary with 'automaton' and 'config' keys")
        return False

    print("\nSuccessfully extracted HA specification from agent output")
    print(f"HA contains {len(ha_specification['automaton'].get('mode', []))} modes and {len(ha_specification['automaton'].get('edge', []))} edges")

    # Find test data file in the input data path
    test_data_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_data_files:
        print(f"\nWarning: No .npz test data files found in {input_data_path}")
        return False

    # Use first test file found
    npz_file_path = os.path.join(input_data_path, test_data_files[0])
    print(f"\nUsing test data file: {npz_file_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create evaluator
        evaluator = HAEvaluator(
            ha_dict=ha_specification,
            npz_file_path=npz_file_path,
            dt=ha_specification['config'].get('dt', 0.001),
            total_time=ha_specification['config'].get('total_time', 10.0)
        )

        # Run evaluation with overlay plot
        save_path = os.path.join(output_dir, 'ha_evaluation_comparison.png')

        print("\nRunning HA evaluation...")
        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path=save_path,
            print_metrics=True
        )

        print(f"\nEvaluation complete! Results saved to: {save_path}")

        # Save metrics and HA specification to file
        metrics_file = os.path.join(output_dir, 'ha_evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Metrics\n")
            f.write("=" * 80 + "\n\n")
            f.write(metrics_text)
            f.write("\n\nHA Specification:\n")
            f.write(json.dumps(ha_specification, indent=2))
        print(f"Metrics saved to: {metrics_file}")

        return True

    except Exception as e:
        print(f"\nError during HA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        default=["hybrid_automaton_image_analysis", "ask_review_expert_ha", "summarize_ha_iterations"],
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
    ap.add_argument(
        "--summarize-tool-model",
        type=str,
        default="gemini/gemini-2.5-flash-lite",
        help="Model ID to use for the summarize iterations tool (Gemini API format).",
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
        summarize_tool_model=args.summarize_tool_model,
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
        managed_agents_list_model_id=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else None,
        manager_type=args.manager_type,
        tools_to_remove=args.tools_to_remove,
    )

    # Obtain task and images
    task, compressed_trace_images = obtain_task_and_images(
        input_data_path=args.input_data_path,
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

    # Run the agent with task and compressed images
    result = managerAgent.run(task, images=compressed_trace_images)

    # Evaluate the generated HA specification
    evaluate_ha_specification(result, args.input_data_path)


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_llm_ha_beta.py --input-data-path utils/Dainarx_code/data_duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis ask_review_expert_ha
