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
from utils.validateTools_ha import ValidateHASpecTool
import numpy as np


def get_data_dimensions(input_data_path: str) -> tuple[int, int]:
    """
    Read the .npz data file and extract the number of state variables and inputs.
    
    Args:
        input_data_path: Path to the directory containing .npz files
        
    Returns:
        Tuple of (num_variables, num_inputs)
    """
    # Find first .npz file in the directory
    npz_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_data_path}")
    
    npz_path = os.path.join(input_data_path, npz_files[0])
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract dimensions from data shape
    # state: shape=(num_vars, num_steps), input: shape=(num_inputs, num_steps)
    num_variables = data['state'].shape[0]
    num_inputs = data['input'].shape[0] if 'input' in data and data['input'].size > 0 else 0
    
    print(f"Auto-detected from data: num_variables={num_variables}, num_inputs={num_inputs}")
    return num_variables, num_inputs


def generate_dynamic_ha_template(num_variables: int, num_inputs: int, system_name: str = "Unknown System") -> str:
    """
    Generate a dynamic HA specification template with pre-filled var and input fields.
    
    Args:
        num_variables: Number of state variables
        num_inputs: Number of input variables
        system_name: Name of the system for comments
        
    Returns:
        JSON string template with correct var and input fields
    """
    # Generate variable names: x1, x2, ..., xN
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    
    # Generate input names: u1, u2, ..., uM (or empty string if no inputs)
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else ""
    
    # Generate placeholder equation based on number of variables
    if num_variables == 1:
        # Single variable - use 2nd-order ODE format
        eq_placeholder = "x1[2] = -0.5 * x1[1] - 5.0 * x1[0]"
        if num_inputs > 0:
            eq_placeholder += " + u1"
        dim = 2  # 2nd-order system
    else:
        # Multiple variables - use 1st-order ODE format
        eq_parts = []
        for i in range(num_variables):
            var = f"x{i+1}"
            # Simple placeholder: dx_i/dt = sum of other variables
            terms = [f"-0.5 * {var}[0]"]
            if i < num_variables - 1:
                terms.append(f"x{i+2}[0]")
            eq_parts.append(f"{var}[1] = {' + '.join(terms)}")
        if num_inputs > 0:
            eq_parts[-1] += " + u1"
        eq_placeholder = ", ".join(eq_parts)
        dim = 1  # 1st-order system
    
    template = f'''{{
    "automaton": {{
        "var": "{var_names}",
        "input": "{input_names}",
        "mode": [
            {{
                "id": 1,
                "eq": "{eq_placeholder}"
            }}
        ],
        "edge": []
    }},
    "config": {{
        "dt": 0.001,
        "total_time": 10.0,
        "dim": {dim},
        "need_reset": false,
        "non_linear_items": ""
    }}
}}'''
    return template


# Tool mapping for HA-specific tools
TOOLNAME2TOOL = {
    'hybrid_automaton_image_analysis': HybridAutomatonImageTool,## if the managerAgent can not input image, use this tool
    'hybrid_automaton_review_expert': ReviewRequestTool_ha,
    'summarize_hybrid_automaton_iterations': SummarizeMemoryTool,
    'validate_hybrid_automaton_specification': ValidateHASpecTool,
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
                           manager_type: str = "CodeAgent",
                           feedback: str = None, # Added feedback parameter
                           iteration: int = 1) -> tuple[str, list]:
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
        feedback: Feedback string from previous iteration (optional)
        iteration: Current iteration number (1-indexed)

    Returns:
        Tuple of (task prompt string, list of compressed images)
    '''
    print(f"Generating task for iteration {iteration}, tools_list: {tools_list}")

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

## Analysis Workflow \n"""

    # Add tool-specific prompts
    if HybridAutomatonImageTool in ToolsList:
        HA_IMAGE_TOOL_PROMPT = "You MUST use the 'hybrid_automaton_image_analysis' tool to analyze the image."
    REVIEW_TOOL_PROMPT = " When you need expert review of your hybrid automaton specification, you MUST call the `hybrid_automaton_review_expert` tool."
    if ReviewRequestTool_ha in ToolsList:
        REVIEW_TOOL_PROMPT += """
**MANDATORY BEFORE FINALIZATION**: You MUST call `hybrid_automaton_review_expert` at least once before submitting your final answer to ensure specification correctness and completeness."""

    VALIDATE_TOOL_PROMPT = ""
    if ValidateHASpecTool in ToolsList:
        VALIDATE_TOOL_PROMPT = """
**VALIDATION TOOL**: Before submitting your final answer, you SHOULD call `validate_hybrid_automaton_specification` to check for syntax errors.
This tool will:
- Detect common formatting issues (wrong direction format, missing required fields, etc.)
- Auto-fix minor issues and normalize the specification
- Report critical errors that need manual fixing

⚠️ **IMPORTANT**: If validation returns a FIXED specification, use the corrected version in your final answer!"""

    task += HA_IMAGE_TOOL_PROMPT if HybridAutomatonImageTool in ToolsList else ""
    task += REVIEW_TOOL_PROMPT if ReviewRequestTool_ha in ToolsList else ""
    task += VALIDATE_TOOL_PROMPT

    task += """

## HA Refinement Guidelines
Adopt a **Parsimonious Modeling Approach** (Occam's Razor) - favor simpler explanations unless the data demands otherwise:

1. **Symbolic Identification**:
   - **Hypothesize Structure First**: Determine the likely functional form of the equations *before* estimating parameters.

2. **Mode Count Strategy (Less is More)**:
   - **Iterative Expansion**: Start with **1 Mode**. If a single continuous model fails to fit the entire trajectory (high error), try **2 Modes**, etc.
   - **Hypothesis Testing**: Increase the number of modes *only* if distinct switching behaviors (sharp changes in dynamics) are observed.

3. **Mode Dynamics**:
   - **Start Simple**: Attempt to fit **Linear** dynamics first.
   - **Increase Complexity**: Only introduce **Non-linear** terms (polynomial, trigonometric, etc.) if linear fits fail to capture curvature or key features.

4. **Guard Conditions (Geometric Simplicity)**:
   - **Single-Variable Thresholds**: The vast majority of physical guards are simple threshold checks on a single variable (e.g., `x1 >= 0`, `x <= 0.5`).
   - **Avoid Overfitting**: Do not create complex arithmetic guards (e.g., `x1*x2 > 5`) unless the physical interaction explicitly suggests it.
   - **Operators**: Stick to standard comparison operators (`<=`, `>=`).

5. **Transition Structure**:
   - **Topology**: Prefer **Sparse** connectivity. Valid transitions are typically few and distinct.
   - **Flow**: Transitions often follow a logical flow (e.g., cycles, bidirectional switches) rather than random jumps.

6. **Reset Maps**:
   - **Continuity Default**: Assume physical variables change **Continuously** (`need_reset: false`) over time.
   - **Exceptions**: Use reset maps (`need_reset: true`) **only** if the data clearly shows instantaneous state jumps at transition points.

7. **Structure Validation**:
   - Ensure the number of modes matches the distinct behaviors observed.
   - Ensure guards partition the state space logically (no overlapping active modes usually).

## Quality Criteria
Your HA specification will be evaluated on:
- **Trajectory Matching**: Simulated output should closely follow ground truth data
- **Mode Detection Accuracy**: Correct identification of switching instants (`tc` metric)
- **State Error Minimization**: Low `mean_diff` and `max_diff` between predicted and actual states"""

    # Add output requirements with clear format specification
#     task += """

# ## Output Format Requirements
# Return a **valid JSON object** with the following structure:
# ```json
# {
#     "automaton": {
#         "var": "x1, x2, ...",
#         "input": "u1, u2, ...",
#         "mode": [{"id": 1, "eq": "..."}],
#         "edge": [{"direction": "1 -> 2", "condition": "...", "reset": {...}}]
#     },
#     "config": {
#         "dt": 0.001,
#         "total_time": 10.0,
#         "dim": 1,
#         "need_reset": true,
#         "non_linear_items": "..."
#     }
# }
# ```
# **CRITICAL JSON RULES**:
# - Use `true`/`false` (NOT Python's `True`/`False`)
# - Use double quotes `"key"` (NOT single quotes)
# - Return ONLY the JSON. No markdown formatting, no explanations, no code blocks in the final answer."""

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

    # Generate dynamic HA template with correct var and input fields pre-filled
    dynamic_ha_template = generate_dynamic_ha_template(num_variables, num_inputs, system_name)
    
    # Generate variable and input names for display
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else "(none)"

    # Add HA specification format documentation and initial spec
    task += f"""
{HA_SPEC_DOCUMENTATION}

### ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification Template (v0 - with correct dimensions)
The `var` and `input` fields are pre-filled. Your task is to refine the **equations** and **structure**:

```json
{dynamic_ha_template}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "{var_names}"` and `input: "{input_names if num_inputs > 0 else ''}"` exactly as shown!**
- Refine mode equations to match observed dynamics
- Add modes and edges if switching behavior is detected
"""

    # Add feedback from previous iteration if available
    if feedback:
        task += f"""
    
    ## ⚠️ FEEDBACK FROM PREVIOUS ITERATION
    The following feedback was generated from evaluating your previous attempt. Use it to guide your next refinement:
    
    {feedback}
    
    """

    # Add trace data description
#     task += f"""

# ## Observed Trace Data
# {trace_data_text}
# """

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
    
    # Import HA specification validator
    from utils.ha_spec_validator import preprocess_ha_for_evaluation

    # Use the validator to extract and fix HA specification
    print("\n--- HA Specification Validation ---")
    ha_specification, is_valid, validation_message = preprocess_ha_for_evaluation(agent_result)
    print(validation_message)
    
    if not is_valid:
        print("\nError: HA specification validation failed")
        if ha_specification is not None:
            print(f"Partially extracted spec: {json.dumps(ha_specification, indent=2)[:500]}...")
        return False

    # Final structure check (should pass after validation, but double-check)
    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        print("\nWarning: Could not extract valid HA specification from agent output for evaluation")
        print("Expected dictionary with 'automaton' and 'config' keys")
        return False

    print("\n--- HA Specification Summary ---")
    print(f"Variables: {ha_specification['automaton'].get('var', 'N/A')}")
    print(f"Inputs: {ha_specification['automaton'].get('input', 'N/A')}")
    print(f"Modes: {len(ha_specification['automaton'].get('mode', []))}")
    print(f"Edges: {len(ha_specification['automaton'].get('edge', []))}")

    # Find test data file in the input data path
    test_data_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_data_files:
        print(f"\nWarning: No .npz test data files found in {input_data_path}")
        return False

    # Use first test file found
    npz_file_path = os.path.join(input_data_path, test_data_files[0])
    print(f"\nUsing test data file: {npz_file_path}")

    # Load ground truth data to check dimensions
    gt_data = np.load(npz_file_path, allow_pickle=True)
    gt_num_vars = gt_data['state'].shape[0]
    
    # Count variables in HA spec
    var_str = ha_specification['automaton'].get('var', '')
    ha_num_vars = len([v.strip() for v in var_str.split(',') if v.strip()])
    
    print(f"\n--- Dimension Check ---")
    print(f"Ground truth state variables: {gt_num_vars}")
    print(f"HA specification variables: {ha_num_vars} ({var_str})")
    
    if ha_num_vars != gt_num_vars:
        print(f"\n⚠️ ERROR: Variable count mismatch!")
        print(f"HA spec defines {ha_num_vars} variable(s), but ground truth has {gt_num_vars} variable(s).")
        print(f"The LLM may have incorrectly converted to state-space form.")
        print(f"For a {gt_num_vars}-variable system, use higher-order ODEs (e.g., x1[2] = ...) instead of multiple 1st-order ODEs.")
        return False

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


def evaluate_ha_specification_with_feedback(agent_result, input_data_path: str, output_dir: str = None) -> tuple[bool, dict, str]:
    """
    Evaluate the generated Hybrid Automaton specification and return feedback for the agent.

    Args:
        agent_result: Result from the agent.run() call
        input_data_path: Path to the directory containing test .npz files
        output_dir: Directory to save evaluation results

    Returns:
        Tuple of (success_bool, metrics_dict, feedback_string)
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Testing the generated Hybrid Automaton specification")
    print("=" * 80)

    # Import HA evaluation module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'Dainarx_code'))
    from HA_evaluation import HAEvaluator
    
    # Import HA specification validator
    from utils.ha_spec_validator import preprocess_ha_for_evaluation

    # Use the validator to extract and fix HA specification
    print("\n--- HA Specification Validation ---")
    ha_specification, is_valid, validation_message = preprocess_ha_for_evaluation(agent_result)
    print(validation_message)
    
    if not is_valid:
        error_msg = "HA specification validation failed. " + validation_message
        if ha_specification is not None:
             error_msg += f"\nPartially extracted spec: {json.dumps(ha_specification, indent=2)[:500]}..."
        return False, {}, error_msg

    # Final structure check
    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        return False, {}, "Could not extract valid HA specification from agent output (missing 'automaton' or 'config')."

    # Find test data file
    test_data_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_data_files:
        return False, {}, f"No .npz test data files found in {input_data_path}"

    npz_file_path = os.path.join(input_data_path, test_data_files[0])
    
    # Check dimensions
    gt_data = np.load(npz_file_path, allow_pickle=True)
    gt_num_vars = gt_data['state'].shape[0]
    var_str = ha_specification['automaton'].get('var', '')
    ha_num_vars = len([v.strip() for v in var_str.split(',') if v.strip()])
    
    if ha_num_vars != gt_num_vars:
        msg = f"Variable count mismatch! HA spec has {ha_num_vars}, ground truth has {gt_num_vars}. Check state-space vs higher-order ODE format."
        return False, {}, msg

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

        # Run evaluation - use absolute path to avoid path conversion in HAEvaluator
        save_num = os.urandom(4).hex()
        save_path = os.path.abspath(os.path.join(output_dir, f'ha_eval_{save_num}.png'))
        metrics_text, plot_base64 = evaluator(
            plot_mode='overlay',
            save_path=save_path,
            print_metrics=True
        )
        metrics_dict = evaluator.metrics

        # Save metrics
        metrics_file = os.path.join(output_dir, f'metrics_{save_num}.txt')
        with open(metrics_file, 'w') as f:
            f.write(metrics_text)
            f.write("\n\nHA Specification:\n")
            f.write(json.dumps(ha_specification, indent=2))

        # Construct feedback string is the same from metrics_file
        
        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        feedback += f"Evaluation Results:\n{metrics_text}\n"


        return True, metrics_dict, feedback

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, {}, f"Error during simulation/evaluation: {str(e)}"


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
        default=["hybrid_automaton_image_analysis", "summarize_hybrid_automaton_iterations", "validate_hybrid_automaton_specification"],
        help="List of tool names to use in the agent.",
    )# not include hybrid_automaton_review_expert for fast output final answer

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

    ap.add_argument(
        "--tools-to-remove",
        type=str,
        nargs='*',
        default=['web_search', 'visit_webpage'],
        help="List of default tools to remove from the agent.",
    )

    # HA-Scientist Loop parameters
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations (HA-Scientist loop). Default: 3",
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

    # Auto-detect dimensions from data file (overrides command-line args if provided)
    auto_num_variables, auto_num_inputs = get_data_dimensions(args.input_data_path)
    
    # Use auto-detected values (can be overridden by explicit command-line args if needed)
    # If user explicitly provided different values, warn them
    if args.num_variables != auto_num_variables:
        print(f"⚠️ Warning: --num-variables={args.num_variables} differs from auto-detected value={auto_num_variables}")
        print(f"   Using auto-detected value: {auto_num_variables}")
    if args.num_inputs != auto_num_inputs:
        print(f"⚠️ Warning: --num-inputs={args.num_inputs} differs from auto-detected value={auto_num_inputs}")
        print(f"   Using auto-detected value: {auto_num_inputs}")
    
    num_variables = auto_num_variables
    num_inputs = auto_num_inputs

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

    # HA-Scientist Iterative Loop
    best_result = None
    best_error = float('inf')
    current_feedback = ""

    print(f"\nSTARTING SR-SCIENTIST LOOP (Max iterations: {args.max_iterations})")
    
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*40}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'#'*40}")

        # Obtain task and images with feedback from previous iteration
        task, compressed_trace_images = obtain_task_and_images(
            input_data_path=args.input_data_path,
            system_name=args.system_name,
            num_variables=num_variables,
            num_inputs=num_inputs,
            initial_ha_spec=None,
            tools_list=args.tools_list,
            managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
            manager_type=args.manager_type,
            feedback=current_feedback,
            iteration=iteration
        )
        
        # save the task to a file
        task_filename = f"task_iter_{iteration-1}.md"
        with open(task_filename, "w", encoding="utf-8") as f:
            f.write(task)
        print(f"Saved task to {task_filename}")

        # Run the agent with task and compressed images
        try:
            result = managerAgent.run(task, images=compressed_trace_images)
        except Exception as e:
            print(f"Agent execution failed: {e}")
            current_feedback = f"Agent execution failed in previous iteration: {str(e)}. Please try to generate a valid specification."
            continue

        # Evaluate the generated HA specification
        success, metrics, feedback_str = evaluate_ha_specification_with_feedback(
            result, 
            args.input_data_path,
            output_dir=os.path.join("evaluation_results", f"iter_{iteration}")
        )
        
        current_feedback += 'Hybrid Automaton Specification v' + str(iteration) + ':\n' + feedback_str # Update feedback for next loop

        # Check if this is the best result so far
        # We need a metric to minimize. Let's assume 'rmse' or similar is in metrics dict.
        # If metrics is empty or parsing failed, we treat error as infinite.
        
        # Currently HAEvaluator might not return a clean 'error' float in dictionary unless we parse the text or modify HAEvaluator.
        # For now, we rely on the feedback string being generated. 
        # But to track "best", we need a scalar.
        # HAEvaluator returns (text, dict). Let's assume dict has 'mean_diff' or similar.
        
        current_error = float('inf')
        if success and isinstance(metrics, dict):
            # Try to find an error metric
            if 'mean_diff' in metrics:
                current_error = metrics['mean_diff']
            elif 'rmse' in metrics:
                current_error = metrics['rmse']
        
        print(f"Iteration {iteration} Result Error: {current_error}")

        if success and current_error < best_error:
            best_error = current_error
            best_result = result
            print(f"New Best Result Found! (Error: {best_error})")
        
        # Optional: Early stopping if error is sufficiently low
        if best_error < 0.01: # Example threshold
            print("Target accuracy reached. Stopping early.")
            break

    print("\n" + "="*80)
    print("HA-SCIENTIST LOOP COMPLETE")
    print(f"Best Error Achieved: {best_error}")
    print("="*80)

    # Final Evaluation of the best result (saved to main evaluation folder)
    if best_result:
        evaluate_ha_specification_with_feedback(best_result, args.input_data_path, output_dir="evaluation_results")


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_llm_ha_beta.py --input-data-path utils/Dainarx_code/data_duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis summarize_hybrid_automaton_iterations validate_hybrid_automaton_specification
