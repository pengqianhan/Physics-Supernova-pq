"""
run_dainarx.py - LLM Agent for Dynamic Hybrid Automaton Generation and Validation

This script enables the ManagerAgent to:
1. Dynamically generate Hybrid Automaton JSON specifications
2. Validate them in real-time using the DainarxEvaluationTool
3. Receive immediate feedback (metrics) from Dainarx validation
4. Iteratively refine specifications until quality targets are met

Key Features:
- No dependency on pre-written JSON templates
- Real-time validation feedback during agent execution
- Multi-iteration refinement loop with intelligent feedback selection
- Integration with Dainarx HAEvaluator for trajectory comparison

Based on: run_llm_ha_beta.py
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("dotenv loaded successfully from .env file")
    print(f"Currently using api key: {os.environ.get('GEMINI_API_KEY', 'Not Set')[:20]}...")
except ImportError:
    print("dotenv not available, continuing without it")
except Exception:
    print("Error loading .env file, continuing without it")

import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

from smolagents.default_tools import Tool
from smolagents import (
    MultiStepAgent,
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)

# Import markdown utilities
from utils import MarkdownMessage
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress
from utils.utils import HAHyperparameters, IterationResult, ResultsAggregator

# Import HA-specific tools
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.reviewTools_ha import ReviewRequestTool_ha
from utils.summemoryTools_ha import SummarizeMemoryTool
from utils.validateTools_ha import ValidateHASpecTool
from utils.dainarxTools import DainarxEvaluationTool, evaluate_ha_specification

# Import prompts
from prompts_ha.prompts import (
    HA_SPEC_DOCUMENTATION,
    get_ha_spec_documentation_with_schema,
)


def get_data_dimensions(input_data_path: str) -> tuple[int, int]:
    """
    Read the .npz data file and extract the number of state variables and inputs.
    """
    npz_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_data_path}")

    npz_path = os.path.join(input_data_path, npz_files[0])
    data = np.load(npz_path, allow_pickle=True)

    num_variables = data['state'].shape[0]
    num_inputs = data['input'].shape[0] if 'input' in data and data['input'].size > 0 else 0

    print(f"Auto-detected from data: num_variables={num_variables}, num_inputs={num_inputs}")
    return num_variables, num_inputs


def generate_dynamic_ha_template(num_variables: int, num_inputs: int, system_name: str = "Unknown System") -> str:
    """
    Generate a dynamic HA specification template with pre-filled var and input fields.
    """
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else ""

    if num_variables == 1:
        eq_placeholder = "x1[2] = -0.5 * x1[1] - 5.0 * x1[0]"
        if num_inputs > 0:
            eq_placeholder += " + u1"
        order = 2
    else:
        eq_parts = []
        for i in range(num_variables):
            var = f"x{i+1}"
            terms = [f"-0.5 * {var}[0]"]
            if i < num_variables - 1:
                terms.append(f"x{i+2}[0]")
            eq_parts.append(f"{var}[1] = {' + '.join(terms)}")
        if num_inputs > 0:
            eq_parts[-1] += " + u1"
        eq_placeholder = ", ".join(eq_parts)
        order = 1

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
        "order": {order},
        "need_reset": false,
        "non_linear_items": ""
    }}
}}'''
    return template


# Tool mapping for HA-specific tools (including new DainarxEvaluationTool)
TOOLNAME2TOOL = {
    'hybrid_automaton_image_analysis': HybridAutomatonImageTool,
    'hybrid_automaton_review_expert': ReviewRequestTool_ha,
    'summarize_hybrid_automaton_iterations': SummarizeMemoryTool,
    'validate_hybrid_automaton_specification': ValidateHASpecTool,
    'dainarx_evaluate_ha': DainarxEvaluationTool,  # NEW: Direct Dainarx validation
}


def _create_HA_agent(Tools_list: List[type[Tool]],
                     markdown_content: MarkdownMessage,
                     model_id: str = "gemini/gemini-flash-lite-latest",
                     managed_agents_list: List[MultiStepAgent] = None,
                     max_steps: int = 80,
                     input_data_path: str = None,
                     **kwargs) -> ToolCallingAgent | CodeAgent:
    """
    Create the HA learning agent with tools including the DainarxEvaluationTool.
    """
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200,
        thinking_level="low"
    )

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
        elif tool.__name__ == "DainarxEvaluationTool":
            # Pass data path to DainarxEvaluationTool
            tools.append(tool(data_path=input_data_path))
        else:
            tools.append(tool())

    manager_agent_kwargs = dict(
        model=model,
        tools=tools,
        max_steps=max_steps,
        verbosity_level=2,
        name="ha_dainarx_agent",
        description="Agent for dynamic HA generation and Dainarx validation",
        managed_agents=managed_agents_list,
        add_base_tools=False
    )

    if kwargs["manager_type"] == "CodeAgent":
        manager_agent_kwargs["additional_authorized_imports"] = [
            "os", "sys", "time", "argparse", "pathlib",
            "matplotlib.pyplot", "matplotlib", "pandas", "json",
            "numpy", "numpy.linalg", "numpy.fft", "numpy.random",
            "numpy.polynomial", "numpy.ma", "numpy.lib",
            "scipy", "scipy.linalg", "scipy.optimize", "scipy.interpolate",
            "scipy.integrate", "scipy.stats", "scipy.signal", "scipy.fft",
            "scipy.sparse", "scipy.ndimage", "scipy.special"
        ]
        managerAgent = CodeAgent(**manager_agent_kwargs)
    elif kwargs["manager_type"] == "ToolCallingAgent":
        manager_agent_kwargs["max_tool_threads"] = 1
        managerAgent = ToolCallingAgent(**manager_agent_kwargs)
    else:
        raise ValueError(f"Unknown manager type: {kwargs['manager_type']}")

    # Inject agent reference into each tool (delayed injection pattern)
    for toolName in managerAgent.tools:
        managerAgent.tools[toolName].worker_agent = managerAgent

    # Store high-res images in agent for tools to access
    managerAgent.markdown_content_high_res_image = markdown_content

    return managerAgent


def get_managed_agents_list(managed_agents_list: List[str] = None,
                            managed_agents_list_model_id: str = None,
                            input_data_path: str = None) -> List[MultiStepAgent]:
    """Create managed agents for numerical analysis tasks."""
    if managed_agents_list is None:
        return []

    managed_agents = []
    for agent_name in managed_agents_list:
        model = LiteLLMModel(
            model_id=managed_agents_list_model_id,
            api_key=os.environ.get("GEMINI_API_KEY"),
            max_completion_tokens=24576,
            num_retries=3,
            timeout=1200
        )

        # Use provided data path or default
        trace_file_path = os.path.join(input_data_path, "sample_train_0.npz") if input_data_path else \
            os.path.join(os.path.dirname(__file__), "utils", "Dainarx_code", "data_duffing", "sample_train_0.npz")

        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can assist with code-related tasks. The data file path is {trace_file_path}.

CAPABILITIES:
1. **Numerical Analysis**: I can load .npz data and perform numpy/scipy operations.
2. **Curve Fitting**: I can fit linear/nonlinear models to data segments.
3. **Windowed Error Analysis (Mode Detection)**: I can detect hidden mode switches in smooth data using this workflow:
   - Define a window size (e.g., 10 steps).
   - Slide the window across the trajectory.
   - In each window, fit a simple local model (e.g., linear dx/dt = Ax).
   - Plot/analyze the fitting residual (error) or parameter variation over time.
   - **Insight**: Spikes in residual or jumps in parameters indicate a MODE SWITCH, even if the curve looks smooth.
"""
        use_e2b = bool(os.environ.get("E2B_API_KEY"))
        managed_agent = CodeAgent(
            tools=[],
            executor_type="e2b" if use_e2b else "python",
            model=model,
            name=agent_name,
            additional_authorized_imports=[
                "os", "sys", "time", "argparse", "pathlib",
                "matplotlib.pyplot", "matplotlib", "pandas", "json",
                "numpy", "numpy.linalg", "numpy.fft", "numpy.random",
                "numpy.polynomial", "numpy.ma", "numpy.lib",
                "scipy", "scipy.linalg", "scipy.optimize", "scipy.interpolate",
                "scipy.integrate", "scipy.stats", "scipy.signal", "scipy.fft",
                "scipy.sparse", "scipy.ndimage", "scipy.special"
            ],
            description=managed_agent_description,
            max_steps=80,
            verbosity_level=2,
        )
        managed_agents.append(managed_agent)

    return managed_agents


def create_agent(model_id: str = "gemini/gemini-flash-lite-latest",
                 input_data_path: str = None,
                 tools_list: List[str] = [],
                 managed_agents_list: List[str] = None,
                 managed_agents_list_model_id: str = None,
                 **kwargs) -> ToolCallingAgent | CodeAgent:
    """Create a Hybrid Automaton learning agent with Dainarx validation capabilities."""
    markdown_content = load_trace_data_from_filepath(input_data_path)
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]

    haAgent = _create_HA_agent(
        Tools_list=ToolsList,
        markdown_content=markdown_content,
        model_id=model_id,
        managed_agents_list=get_managed_agents_list(managed_agents_list, managed_agents_list_model_id, input_data_path),
        input_data_path=input_data_path,
        **kwargs
    )
    return haAgent


def obtain_task_and_images(input_data_path: str = None,
                           system_name: str = "Unknown System",
                           num_variables: int = 1,
                           num_inputs: int = 1,
                           tools_list: List[str] = [],
                           managed_agents_list: List[str] = None,
                           manager_type: str = "CodeAgent",
                           feedback: str = None,
                           iteration: int = 1,
                           use_json_schema: bool = True) -> tuple[str, list]:
    """
    Generate task prompt and compressed images for HA learning agent.

    This version emphasizes the use of dainarx_evaluate_ha tool for real-time validation.
    """
    print(f"Generating task for iteration {iteration}, tools_list: {tools_list}")

    markdown_content = load_trace_data_from_filepath(input_data_path)
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]
    trace_data_text = markdown_to_plaintext(markdown_content)
    compressed_trace_images = markdown_images_compress(markdown_content, max_short_side_pixels=1080)

    # Base task prompt
    task = f"""# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK (DAINARX MODE)

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## KEY CAPABILITY: Real-Time Validation with Dainarx
You have access to the `dainarx_evaluate_ha` tool that allows you to **validate your HA specification in real-time**.

**WORKFLOW:**
1. Analyze the trajectory images to understand system dynamics
2. Generate an HA JSON specification
3. **IMMEDIATELY** call `dainarx_evaluate_ha` with your specification
4. Review the metrics (TC, Max Diff, Mean Diff) returned by Dainarx
5. If metrics are poor, refine your specification and re-validate
6. Iterate until you achieve good metrics (Max Diff < 0.005)

## Problem Context
You are given time-series trajectory data from an unknown hybrid dynamical system. Your task is to:
1. **Identify discrete modes** (operating regimes with distinct continuous dynamics)
2. **Infer mode-specific ODEs** (differential equations governing each regime)
3. **Determine switching conditions** (guard predicates triggering mode transitions)
4. **Validate against ground truth** using the `dainarx_evaluate_ha` tool

## Available Data
The following trace data visualizations are provided (reference images using placeholders: `<image_0>`, `<image_1>`, etc.):
- State variable trajectories over time
- Input signals (if applicable)
- Potential mode-switch indicators (discontinuities, slope changes)

## Analysis Workflow
1. **Visual Mode Analysis (MANDATORY)**:
   - Use the `hybrid_automaton_image_analysis` tool to inspect the trajectory images
   - Identify the number of modes and their approximate time intervals

2. **Generate Initial HA Specification**:
   - Based on visual analysis, create an HA JSON specification
   - Start simple (single mode) and add complexity only if needed

3. **VALIDATE WITH DAINARX (CRITICAL)**:
   - Call `dainarx_evaluate_ha` with your JSON specification
   - This will simulate your HA and compare against ground truth
   - Review the metrics returned:
     - TC < 0.01s is good, TC < 0.001s is excellent
     - Max Diff < 0.005 is good, Max Diff < 0.0001 is excellent

4. **Iterative Refinement**:
   - If metrics are poor, analyze the feedback
   - Adjust ODE parameters, add modes, or modify guards
   - Re-validate with `dainarx_evaluate_ha`
   - Repeat until metrics meet quality targets

## HA Refinement Guidelines
1. **Start Simple**: Begin with 1 mode and linear dynamics
2. **Add Complexity Gradually**: Only add modes/nonlinearities if metrics don't improve
3. **Use Dainarx Feedback**: The `dainarx_evaluate_ha` tool gives you exact error metrics
4. **Guard Conditions**: Use simple thresholds (e.g., `x1 >= 0`, `x1 <= 0.5`)

## Quality Criteria (Lower is better)
- **TC (Change-Point Error)**: < 0.01s is good, <= 0.001s is final destination
- **Max Difference**: < 0.005 is good, <= 0.0001 is final destination
- **Mean Difference**: < 0.005 is good, <= 0.00001 is final destination"""

    # Add tool-specific prompts
    HA_IMAGE_TOOL_PROMPT = ""
    if HybridAutomatonImageTool in ToolsList:
        HA_IMAGE_TOOL_PROMPT = "\n\n**Image Analysis**: Use `hybrid_automaton_image_analysis` to analyze the trajectory images."

    DAINARX_TOOL_PROMPT = ""
    if DainarxEvaluationTool in ToolsList:
        DAINARX_TOOL_PROMPT = """

**CRITICAL - DAINARX VALIDATION**:
You MUST use the `dainarx_evaluate_ha` tool to validate EVERY HA specification you generate.
This tool will simulate your HA and return quantitative metrics showing how well it matches the ground truth.
Do NOT submit a final answer without first validating it with Dainarx!"""

    VALIDATE_TOOL_PROMPT = ""
    if ValidateHASpecTool in ToolsList:
        VALIDATE_TOOL_PROMPT = """

**Syntax Validation**: Before calling `dainarx_evaluate_ha`, you can use `validate_hybrid_automaton_specification` to check for syntax errors."""

    task += HA_IMAGE_TOOL_PROMPT
    task += DAINARX_TOOL_PROMPT
    task += VALIDATE_TOOL_PROMPT

    # Add managed agents prompt
    if managed_agents_list and len(managed_agents_list) > 0:
        MANAGE_AGENT_PROMPT = f"""

## Computational Resources
You have access to managed Code Agent(s): `{managed_agents_list}`
Use them for numerical computations, curve fitting, or complex mathematical derivations."""
        task += MANAGE_AGENT_PROMPT

    if manager_type == "CodeAgent":
        SELF_IS_CODE_AGENT_PROMPT = "\n\n## Code Execution Capability\nYou can use Python Code to execute programs, which may help with your task-solving process."
        task += SELF_IS_CODE_AGENT_PROMPT

    # Generate dynamic HA template
    dynamic_ha_template = generate_dynamic_ha_template(num_variables, num_inputs, system_name)
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else "(none)"

    # Add HA specification format documentation
    if use_json_schema:
        ha_spec_docs = get_ha_spec_documentation_with_schema(simplified=True)
    else:
        ha_spec_docs = HA_SPEC_DOCUMENTATION

    task += f"""
{ha_spec_docs}

### CRITICAL: Variable Count is PRE-DEFINED
The `var` and `input` fields are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- Variables: `{var_names}`
- Inputs: `{input_names if num_inputs > 0 else ''}`

## Initial Template (v0)
Use this as a starting point and refine based on Dainarx validation feedback:

```json
{dynamic_ha_template}
```

## Your Task
1. Analyze the trajectory data
2. Generate an HA specification
3. **VALIDATE with `dainarx_evaluate_ha`** to get metrics
4. Refine based on feedback until Max Diff < 0.005
5. Submit your final validated specification
"""

    # Add feedback from previous iteration if available
    if feedback:
        task += f"""
## FEEDBACK FROM PREVIOUS ITERATION
The following feedback was generated from evaluating your previous attempt:

{feedback}

Use this feedback to guide your next refinement. Focus on improving the metrics.
"""

    return task, compressed_trace_images


def evaluate_ha_specification_with_feedback(
    agent_result,
    input_data_path: str,
    output_dir: str = None,
    hyperparameters: HAHyperparameters = None,
    iteration: int = 1
) -> Tuple[bool, Dict, str, Optional[Dict]]:
    """
    Evaluate the generated HA specification and return feedback.

    This function passes the HA dict directly to Dainarx's main_from_dict() function,
    which is the key integration point for dynamic HA validation.
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Testing the generated Hybrid Automaton specification")
    print("         (Passing HA JSON directly to Dainarx main_from_dict)")
    print("=" * 80)

    # Import Dainarx main_from_dict function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'Dainarx_code'))
    from main import main_from_dict
    from utils.ha_spec_validator import preprocess_ha_for_evaluation

    # Use the validator to extract and fix HA specification
    print("\n--- HA Specification Validation ---")
    ha_specification, is_valid, validation_message = preprocess_ha_for_evaluation(agent_result)
    print(validation_message)

    if not is_valid:
        error_msg = "HA specification validation failed. " + validation_message
        if ha_specification is not None:
            error_msg += f"\nPartially extracted spec: {json.dumps(ha_specification, indent=2)[:500]}..."
        return False, {}, error_msg, ha_specification

    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        return False, {}, "Could not extract valid HA specification (missing 'automaton' or 'config').", None

    # Find test data file for dimension checking
    test_data_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_data_files:
        return False, {}, f"No .npz test data files found in {input_data_path}", ha_specification

    npz_file_path = os.path.join(input_data_path, test_data_files[0])

    # Check dimensions
    gt_data = np.load(npz_file_path, allow_pickle=True)
    gt_num_vars = gt_data['state'].shape[0]
    var_str = ha_specification['automaton'].get('var', '')
    ha_num_vars = len([v.strip() for v in var_str.split(',') if v.strip()])

    if ha_num_vars != gt_num_vars:
        msg = f"Variable count mismatch! HA spec has {ha_num_vars}, ground truth has {gt_num_vars}."
        return False, {}, msg, ha_specification

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # KEY: Call Dainarx main_from_dict() directly with HA dict
        print("\n--- Calling Dainarx main_from_dict() ---")
        eval_results = main_from_dict(
            ha_dict=ha_specification,
            data_path=input_data_path,
            need_creat=False,
            need_plot=False
        )

        # Check for errors
        if 'error' in eval_results:
            feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
            feedback += f"Dainarx Error: {eval_results['error']}"
            return False, {}, feedback, ha_specification

        # Extract metrics from Dainarx results
        metrics_dict = {
            'tc': eval_results.get('tc'),
            'train_tc': eval_results.get('train_tc'),
            'max_diff': eval_results.get('max_diff'),
            'mean_diff': eval_results.get('mean_diff'),
            'clustering_error': eval_results.get('clustering_error'),
        }

        # Build metrics text
        metrics_text = "Dainarx Validation Results (via main_from_dict):\n"
        metrics_text += f"  TC (Change-Point Error): {metrics_dict['tc']:.6f}\n" if metrics_dict['tc'] is not None else "  TC: N/A\n"
        metrics_text += f"  Max Difference: {metrics_dict['max_diff']:.6f}\n" if metrics_dict['max_diff'] is not None else "  Max Diff: N/A\n"
        metrics_text += f"  Mean Difference: {metrics_dict['mean_diff']:.6f}\n" if metrics_dict['mean_diff'] is not None else "  Mean Diff: N/A\n"

        # Save metrics
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(output_dir, f'ha_eval_{timestamp}.txt')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Results (Dainarx main_from_dict)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Iteration: {iteration}\n\n")
            f.write(f"Metrics:\n{metrics_text}\n\n")
            f.write(f"HA Specification:\n{json.dumps(ha_specification, indent=2)}\n\n")
            f.write(f"Raw Dainarx Results:\n{json.dumps(eval_results, indent=2, default=str)}\n\n")
            if hyperparameters is not None:
                f.write(hyperparameters.to_string())

        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        feedback += f"Evaluation Results:\n{metrics_text}\n"

        return True, metrics_dict, feedback, ha_specification

    except Exception as e:
        import traceback
        traceback.print_exc()
        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        feedback += f"main_from_dict() Error: {str(e)}"
        return False, {}, feedback, ha_specification


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "utils", "Dainarx_code", "data_duffing")

    ap = argparse.ArgumentParser(description="Run the HA Learning Agent with Dainarx validation.")

    ap.add_argument("--input-data-path", type=str, default=default_data_path,
                    help="Path to the trace data directory.")
    ap.add_argument("--manager-model", type=str, default="gemini/gemini-flash-lite-latest",
                    help="Model ID to use for the agent.")
    ap.add_argument("--manager-type", type=str, default="CodeAgent",
                    choices=["ToolCallingAgent", "CodeAgent"],
                    help="Type of agent to create.")

    # Tools (including new dainarx_evaluate_ha)
    ap.add_argument("--tools-list", type=str, nargs='*',
                    default=["validate_hybrid_automaton_specification"],
                    # default=["hybrid_automaton_image_analysis", "validate_hybrid_automaton_specification", "dainarx_evaluate_ha"],
                    help="List of tool names to use.")

    # Tool model configurations
    ap.add_argument("--image-tool-model", type=str, default="gemini-flash-lite-latest",
                    help="Model ID for image analysis tool.")
    ap.add_argument("--review-tool-model", type=str, default="gemini-flash-lite-latest",
                    help="Model ID for review tool.")
    ap.add_argument("--summarize-tool-model", type=str, default="gemini/gemini-2.5-flash-lite",
                    help="Model ID for summarize tool.")

    # Managed agents
    ap.add_argument("--managed-agents-list", type=str, nargs='*', default=None,
                    help="List of managed agents.")
    ap.add_argument("--managed-agents-list-model", type=str, default="gemini/gemini-flash-lite-latest",
                    help="Model ID for managed agents.")

    # System configuration
    ap.add_argument("--system-name", type=str, default="Unknown System",
                    help="Name of the dynamical system.")

    # Iteration parameters
    ap.add_argument("--max-iterations", type=int, default=3,
                    help="Maximum number of refinement iterations.")
    ap.add_argument("--feedback-top-k", type=int, default=3,
                    help="Number of top-performing specs to include in feedback.")
    ap.add_argument("--feedback-min-gap", type=float, default=0.005,
                    help="Minimum error gap between selected feedback specs.")
    ap.add_argument("--target-error", type=float, default=0.01,
                    help="Target error threshold for early stopping.")
    ap.add_argument("--no-improvement-patience", type=int, default=3,
                    help="Stop if no improvement for this many iterations.")

    ap.add_argument("--use-json-schema", type=bool, default=True,
                    help="Include JSON Schema in task prompt.")

    args = ap.parse_args()

    if not args.input_data_path:
        raise ValueError("You must provide a data path with --input-data-path.")
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


def main():
    args = parse_args()
    print(f"Running HA Learning Agent (Dainarx Mode)")
    print(f"  Model: {args.manager_model}")
    print(f"  Tools: {args.tools_list}")
    print(f"  Data path: {args.input_data_path}")

    # Auto-detect dimensions
    num_variables, num_inputs = get_data_dimensions(args.input_data_path)

    # Create hyperparameters instance
    managed_agents = args.managed_agents_list if args.managed_agents_list else []
    hyperparameters = HAHyperparameters(
        manager_model=args.manager_model,
        manager_type=args.manager_type,
        managed_agents_count=len(managed_agents),
        managed_agents_list=managed_agents,
        managed_agents_model=args.managed_agents_list_model,
        tools_list=args.tools_list if args.tools_list else [],
        image_tool_model=args.image_tool_model,
        review_tool_model=args.review_tool_model,
        summarize_tool_model=args.summarize_tool_model,
        max_iterations=args.max_iterations,
        feedback_top_k=args.feedback_top_k,
        feedback_min_gap=args.feedback_min_gap,
        target_error=args.target_error,
        no_improvement_patience=args.no_improvement_patience,
        input_data_path=args.input_data_path,
        system_name=args.system_name,
        num_variables=num_variables,
        num_inputs=num_inputs,
        use_json_schema=args.use_json_schema,
    )

    # Create the agent
    managerAgent = create_agent(
        model_id=args.manager_model,
        input_data_path=args.input_data_path,
        tools_list=args.tools_list,
        image_tool_model=args.image_tool_model,
        review_tool_model=args.review_tool_model,
        summarize_tool_model=args.summarize_tool_model,
        managed_agents_list=args.managed_agents_list,
        managed_agents_list_model_id=args.managed_agents_list_model,
        manager_type=args.manager_type,
    )

    # Initialize results aggregator
    results_aggregator = ResultsAggregator(
        top_k=args.feedback_top_k,
        min_gap=args.feedback_min_gap
    )

    print(f"\nSTARTING HA-DAINARX LOOP (Max iterations: {args.max_iterations})")
    print(f"  - Top-K feedback selection: {results_aggregator.top_k}")
    print(f"  - Diversity gap threshold: {results_aggregator.min_gap}")
    print(f"  - Target error for early stop: {args.target_error}")
    print(f"  - No-improvement patience: {args.no_improvement_patience}")

    # Main iteration loop
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*40}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'#'*40}")

        # Generate feedback context
        if iteration == 1:
            current_feedback = ""
        else:
            current_feedback = results_aggregator.get_top_k_feedback()
            latest_feedback = results_aggregator.get_latest_feedback()
            if latest_feedback and latest_feedback not in current_feedback:
                current_feedback += f"\n\n## Most Recent Attempt (Iteration {iteration - 1}):\n{latest_feedback}"

            dynamic_target = results_aggregator.get_dynamic_error_threshold()
            print(f"  [Aggregator] Dynamic error target: {dynamic_target:.6f}")

        # Obtain task and images
        task, compressed_trace_images = obtain_task_and_images(
            input_data_path=args.input_data_path,
            system_name=args.system_name,
            num_variables=num_variables,
            num_inputs=num_inputs,
            tools_list=args.tools_list,
            managed_agents_list=args.managed_agents_list,
            manager_type=args.manager_type,
            feedback=current_feedback,
            iteration=iteration,
            use_json_schema=args.use_json_schema
        )

        # Save task prompt
        task_filename = os.path.join(os.path.dirname(__file__), 'task_prompts', f'dainarx_iter_{iteration}_task.md')
        os.makedirs(os.path.dirname(task_filename), exist_ok=True)
        with open(task_filename, "w", encoding="utf-8") as f:
            f.write(task)

        # Run the agent
        try:
            result = managerAgent.run(task, images=compressed_trace_images)
        except Exception as e:
            print(f"Agent execution failed: {e}")
            failed_result = IterationResult(
                iteration=iteration,
                ha_specification=None,
                metrics={},
                feedback=f"Agent execution failed: {str(e)}",
                success=False,
                error_value=float('inf')
            )
            results_aggregator.add_result(failed_result)
            continue

        # Evaluate the generated HA specification
        success, metrics, feedback_str, ha_spec = evaluate_ha_specification_with_feedback(
            result,
            args.input_data_path,
            output_dir=os.path.join("evaluation_results", f"dainarx_iter_{iteration}"),
            hyperparameters=hyperparameters,
            iteration=iteration
        )

        # Extract error value
        current_error = float('inf')
        if success and isinstance(metrics, dict):
            if 'max_diff' in metrics:
                current_error = metrics['max_diff']
            elif 'mean_diff' in metrics:
                current_error = metrics['mean_diff']
            elif 'tc' in metrics:
                current_error = metrics['tc']

        print(f"Iteration {iteration} Result Error: {current_error}")

        # Store iteration result
        iter_result = IterationResult(
            iteration=iteration,
            ha_specification=ha_spec,
            metrics=metrics if isinstance(metrics, dict) else {},
            feedback=feedback_str,
            success=success,
            error_value=current_error
        )
        results_aggregator.add_result(iter_result)

        # Check for early stopping
        should_stop, stop_reason = results_aggregator.should_early_stop(
            target_error=args.target_error,
            min_iterations=2,
            no_improvement_patience=args.no_improvement_patience
        )

        if should_stop:
            print(f"\n[Early Stop] {stop_reason}")
            break

    # Final Summary
    print("\n" + "="*80)
    print("HA-DAINARX LOOP COMPLETE")
    print("="*80)

    if results_aggregator.best_result:
        print(f"Best Error Achieved: {results_aggregator.best_error:.6f}")
        print(f"Best Iteration: {results_aggregator.best_result.iteration}")

        print("\n--- Iteration Summary ---")
        for res in results_aggregator.results:
            status = "PASS" if res.success else "FAIL"
            error_str = f"{res.error_value:.6f}" if res.error_value < float('inf') else "N/A"
            best_marker = " (BEST)" if res == results_aggregator.best_result else ""
            print(f"  Iter {res.iteration}: [{status}] Error={error_str}{best_marker}")

        # Save best specification
        if results_aggregator.best_result.ha_specification:
            best_spec_path = os.path.join("evaluation_results", "best_ha_specification_dainarx.json")
            os.makedirs("evaluation_results", exist_ok=True)
            with open(best_spec_path, "w", encoding="utf-8") as f:
                json.dump(results_aggregator.best_result.ha_specification, f, indent=2)
            print(f"\nBest HA specification saved to: {best_spec_path}")
    else:
        print("No successful results achieved.")
        print(f"Total iterations attempted: {len(results_aggregator.results)}")


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_dainarx.py --input-data-path utils/Dainarx_code/data_duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis validate_hybrid_automaton_specification dainarx_evaluate_ha
