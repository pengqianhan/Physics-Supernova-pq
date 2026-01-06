import os
import sys
import json
import time
import concurrent.futures
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple
from prompts_ha.prompts import (
    HA_SPEC_DOCUMENTATION,
    get_ha_spec_documentation_with_schema,
)
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
from typing import List, Dict, Tuple, Optional
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
from utils.utils import HAHyperparameters, IterationResult, ResultsAggregator
# Import HA-specific tools
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.reviewTools_ha import ReviewRequestTool_ha
from utils.summemoryTools_ha import SummarizeMemoryTool
from utils.validateTools_ha import ValidateHASpecTool
import numpy as np


# ============================================================================
# Data structures for multi-dataset batch processing
# ============================================================================

class DatasetInfo(NamedTuple):
    """Information about a dataset to process."""
    category: str       # e.g., "ATVA", "FaMoS", "linear", "non_linear"
    name: str           # e.g., "ball", "duffing", "two_tank"
    path: str           # Full path to the dataset directory


@dataclass
class DatasetResult:
    """Result from processing a single dataset."""
    dataset_info: DatasetInfo
    success: bool
    metrics: Dict
    ha_specification: Optional[Dict]
    elapsed_time: float
    error_message: str = ""
    best_iteration: int = 0


# ============================================================================
# Dataset discovery, filtering, and skip logic
# ============================================================================

def discover_datasets(data_all_path: str) -> List[DatasetInfo]:
    """
    Discover all valid datasets in the data_all directory.

    A valid dataset is a directory that contains at least one .npz file.

    Args:
        data_all_path: Path to the data_all directory

    Returns:
        List of DatasetInfo tuples for all valid datasets
    """
    datasets = []

    if not os.path.exists(data_all_path):
        print(f"Warning: data_all path does not exist: {data_all_path}")
        return datasets

    # Iterate through categories (top-level directories)
    for category in sorted(os.listdir(data_all_path)):
        category_path = os.path.join(data_all_path, category)
        if not os.path.isdir(category_path):
            continue

        # Iterate through datasets within each category
        for dataset_name in sorted(os.listdir(category_path)):
            dataset_path = os.path.join(category_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # Check if directory contains .npz files
            npz_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]
            if npz_files:
                datasets.append(DatasetInfo(
                    category=category,
                    name=dataset_name,
                    path=dataset_path
                ))

    print(f"Discovered {len(datasets)} datasets in {data_all_path}")
    return datasets


def filter_datasets(
    datasets: List[DatasetInfo],
    mode: str,
    dataset_names: List[str] = None,
    category_names: List[str] = None
) -> List[DatasetInfo]:
    """
    Filter datasets based on selection mode.

    Args:
        datasets: List of all discovered datasets
        mode: Selection mode: "all", "include", "exclude"
        dataset_names: List of dataset names to include/exclude
        category_names: List of categories to filter by (empty means all)

    Returns:
        Filtered list of DatasetInfo
    """
    dataset_names = dataset_names or []
    category_names = category_names or []

    # First, filter by category if specified
    if category_names:
        datasets = [d for d in datasets if d.category in category_names]

    # Then apply mode-based filtering
    if mode == "all":
        return datasets
    elif mode == "include":
        if not dataset_names:
            print("Warning: --mode include specified but no --datasets provided. Processing all.")
            return datasets
        return [d for d in datasets if d.name in dataset_names]
    elif mode == "exclude":
        if not dataset_names:
            print("Warning: --mode exclude specified but no --datasets provided. Processing all.")
            return datasets
        return [d for d in datasets if d.name not in dataset_names]
    else:
        return datasets


def should_skip_dataset(
    dataset_info: DatasetInfo,
    result_all_path: str,
    skip_existing: bool,
    force_rerun: bool
) -> bool:
    """
    Check if a dataset should be skipped based on existing results.

    Args:
        dataset_info: Information about the dataset
        result_all_path: Path to the result_all directory
        skip_existing: Whether to skip datasets with existing results
        force_rerun: Whether to force rerun (overrides skip_existing)

    Returns:
        True if the dataset should be skipped
    """
    if force_rerun:
        return False

    if not skip_existing:
        return False

    # Check for existing result file
    result_path = os.path.join(
        result_all_path,
        dataset_info.category,
        dataset_info.name,
        "best_ha_specification.json"
    )

    if os.path.exists(result_path):
        print(f"  [SKIP] {dataset_info.category}/{dataset_info.name} - results exist")
        return True

    return False


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
            "matplotlib.pyplot", "matplotlib", "pandas", "json",
            # numpy and all common submodules
            "numpy", "numpy.linalg", "numpy.fft", "numpy.random", 
            "numpy.polynomial", "numpy.ma", "numpy.lib",
            # scipy and all common submodules
            "scipy", "scipy.linalg", "scipy.optimize", "scipy.interpolate",
            "scipy.integrate", "scipy.stats", "scipy.signal", "scipy.fft",
            "scipy.sparse", "scipy.ndimage", "scipy.special"
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

        # trace file path
        trace_file_path = os.path.join(os.path.dirname(__file__), "utils", "Dainarx_code", "data_duffing", "sample_train_0.npz")
        # managed agent
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
        # use_e2b = bool(os.environ.get("E2B_API_KEY"))
        use_e2b = False
        managed_agent = CodeAgent(
            tools=[],
            executor_type="e2b" if use_e2b else "python",
            model=model,
            name=agent_name,
            additional_authorized_imports=[
            "os", "sys", "time", "argparse", "pathlib",
            "matplotlib.pyplot", "matplotlib", "pandas", "json",
            # numpy and all common submodules
            "numpy", "numpy.linalg", "numpy.fft", "numpy.random", 
            "numpy.polynomial", "numpy.ma", "numpy.lib",
            # scipy and all common submodules
            "scipy", "scipy.linalg", "scipy.optimize", "scipy.interpolate",
            "scipy.integrate", "scipy.stats", "scipy.signal", "scipy.fft",
            "scipy.sparse", "scipy.ndimage", "scipy.special"
        ],
            description=managed_agent_description,
            max_steps=80,
            verbosity_level=2,
        )
        if use_e2b:
            print("使用 E2B 云沙盒执行器，正在上传数据文件...")
            # 上传文件到 E2B 沙盒
            with open(trace_file_path, "rb") as f:
                file_content = f.read()
            # E2B 沙盒中的目标路径
            sandbox_file_path = "/tmp/sample_train_0.npz"
            managed_agent.python_executor.sandbox.files.write(sandbox_file_path, file_content)
            print(f"✓ 文件已上传到 E2B 沙盒: {sandbox_file_path}")
            # 更新数据文件路径为沙盒中的路径
            trace_file_path = sandbox_file_path
        else:
            # 本地执行器：将数据文件路径注入到 agent 的状态中
            managed_agent.python_executor.state["DATA_FILE_PATH"] = trace_file_path
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
                           iteration: int = 1,
                           use_json_schema: bool = True) -> tuple[str, list]:
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
        use_json_schema: If True, include JSON Schema in the prompt for structured output

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

## Analysis Workflow
1. **Visual Mode Analysis (MANDATORY)**:
   - **Step 1**: Use the `hybrid_automaton_image_analysis` tool to inspect the trajectory images.
   - **Question to ask**: "How many distinct dynamical regimes (modes) are present? Are there sharp corners, discontinuities, or sudden changes in slope? Return the estimated number of modes and their approximate time intervals."
   - **Decision**: If the image expert reports multiple modes (e.g., "2 regimes", "sharp change at t=5"), you **MUST** proceed with a multi-mode HA structure.

2. **Numerical/Residual Check (Critical for Smooth Systems)**:
   - **Warning**: Some hybrid systems have **smooth trajectories** (no sharp corners) but switch parameters (e.g., stiffness/damping changes). Visual inspection alone may miss this.
   - **Action**: Use the managed agent (if available) or your own Python code to specificially check for changing dynamics.
   - **Logic**: If a single fitted equation has high error in specific time segments, or if the frequency/amplitude decay rate changes noticeably, **you MUST assume multiple modes** even if the curve looks smooth.

3. **Structure Definition**:
   - based on Step 1 & 2, define the number of modes.
   - If >1 mode, define the switching logic (Guards/Transitions).

4. **Parameter Estimation**:
   - Estimate parameters for the ODEs in each mode.

5. **Validation**:
   - Use the `validate_hybrid_automaton_specification` tool to check for syntax errors.

## HA Refinement Guidelines
Adopt a **Parsimonious Modeling Approach** (Occam's Razor) - favor simpler explanations unless the data demands otherwise:

1. **Symbolic Identification**:
   - **Hypothesize Structure First**: Determine the likely functional form of the equations *before* estimating parameters.

2. **Mode Count Strategy (Adaptive Complexity)**:
   - **Visual Evidence Rules**: If the visual analysis (Step 1) detects sharp changes or discontinuities, you **MUST** use multiple modes (2 or more).
   - **Smooth but Hybrid**: If the trajectory is smooth but complex (e.g., varying frequency or damping), prefer a **Switched System** (2+ modes) over a single overly-complex nonlinear equation.
   - **Do NOT Force Single Mode**: Do not attempt to fit a single smooth ODE to data that clearly changes behavior. It is better to have 2 simple linear modes than 1 complex failing non-linear mode.
   - **Template Expansion**: The provided template below is for **1 Mode**. If you detect N modes, you **MUST copy/paste** the mode object to create Mode 2, Mode 3, ... Mode N in your JSON.

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
- **Mode Detection Accuracy**: Correct identification of switching instants (TC (Change-Point Error) < 0.01s is good, <= 0.001s is excellent)
- **State Error Minimization**: Low Mean Difference (Mean Difference) and Maximum Difference (Max Difference) between predicted and actual states (Max Difference < 0.005 is good, < 0.0001 is excellent)"""

    # Add tool-specific prompts (unified style: init empty -> conditionally set -> unconditionally append)
    HA_IMAGE_TOOL_PROMPT = ""
    if HybridAutomatonImageTool in ToolsList:
        HA_IMAGE_TOOL_PROMPT = "\n\nYou MUST use the `hybrid_automaton_image_analysis` tool to analyze the image."

    REVIEW_TOOL_PROMPT = ""
    if ReviewRequestTool_ha in ToolsList:
        REVIEW_TOOL_PROMPT = """
        **MANDATORY BEFORE FINALIZATION**: You MUST call the `hybrid_automaton_review_expert` tool at least once before submitting your final answer to ensure specification correctness and completeness."""

    VALIDATE_TOOL_PROMPT = ""
    if ValidateHASpecTool in ToolsList:
        VALIDATE_TOOL_PROMPT = """
**VALIDATION TOOL**: Before submitting your final answer, you MUST call the `validate_hybrid_automaton_specification` tool to check for syntax and semantic errors.
⚠️ **IMPORTANT**: If validation returns a FIXED specification, use the corrected version in your final answer!"""

    task += HA_IMAGE_TOOL_PROMPT
    task += REVIEW_TOOL_PROMPT
    task += VALIDATE_TOOL_PROMPT


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
    # Use JSON Schema-based documentation for more precise output specification
    if use_json_schema:
        ha_spec_docs = get_ha_spec_documentation_with_schema(simplified=True)
    else:
        ha_spec_docs = HA_SPEC_DOCUMENTATION

    task += f"""
{ha_spec_docs}

### ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification (v0)
The `var` and `input` fields are pre-filled. Your task is to refine the **equations** and **structure**:

```json
{dynamic_ha_template}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "{var_names}"` and `input: "{input_names if num_inputs > 0 else ''}"` exactly as shown!**
- Make sure the HA specification is valid and complete.
- Refine the HA specification to improve trajectory matching and reduce TC (Change-Point Error), Mean Difference, and Maximum Difference.
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


def evaluate_ha_specification_with_feedback(
    agent_result,
    input_data_path: str,
    output_dir: str = None,
    hyperparameters: HAHyperparameters = None,
    iteration: int = 1
) -> Tuple[bool, Dict, str, Optional[Dict]]:
    """
    Evaluate the generated Hybrid Automaton specification and return feedback for the agent.

    Args:
        agent_result: Result from the agent.run() call
        input_data_path: Path to the directory containing test .npz files
        output_dir: Directory to save evaluation results
        hyperparameters: HAHyperparameters instance containing experiment configuration
        iteration: Current iteration number (for logging)

    Returns:
        Tuple of (success_bool, metrics_dict, feedback_string, ha_specification_dict)
        The ha_specification_dict is the extracted/validated HA spec (None if extraction failed)
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
        return False, {}, error_msg, ha_specification

    # Final structure check
    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        return False, {}, "Could not extract valid HA specification from agent output (missing 'automaton' or 'config').", None

    # Find test data file
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
        msg = f"Variable count mismatch! HA spec has {ha_num_vars}, ground truth has {gt_num_vars}. Check state-space vs higher-order ODE format."
        return False, {}, msg, ha_specification

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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.abspath(os.path.join(output_dir, f'ha_eval_{timestamp}.png'))
        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path=save_path,
            print_metrics=True
        )
        metrics_dict = evaluator.metrics

        # Save metrics and hyperparameters to ha_evaluation_metrics.txt
        metrics_file = os.path.join(output_dir, f'ha_eval_{timestamp}.txt')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Iteration: {iteration}\n\n")
            f.write(f"Metrics: \n {metrics_text}\n\n")
            f.write(f"HA Specification: \n{json.dumps(ha_specification, indent=2)}\n\n")

            # Write hyperparameters if provided
            if hyperparameters is not None:
                f.write(hyperparameters.to_string())
                f.write("\n\n")


        # Construct feedback string is the same from metrics_file
        
        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        # feedback = f"```json\n{agent_result}\n```\n"
        feedback += f"Evaluation Results:\n{metrics_text}\n"


        return True, metrics_dict, feedback, ha_specification

    except Exception as e:
        import traceback
        traceback.print_exc()
        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        # feedback = f"```json\n{agent_result}\n```\n"
        feedback += f"Evaluation Results:\nThe HA specification is not valid. Please try to generate a valid specification. Here is the error message: {str(e)}"
        return False, {}, feedback, ha_specification


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
        default="gemini-flash-lite-latest",
        help="Model ID to use for the summarize iterations tool (Gemini API format).",
    )

    # agent names of managed agents
    ap.add_argument(
        "--managed-agents-list",
        type=str,
        nargs='*',
        default=['data_analysis_expert'],
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
        default=[],##['web_search', 'visit_webpage']
        help="List of default tools to remove from the agent.",
    )

    # HA-Scientist Loop parameters
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations (HA-Scientist loop). Default: 3",
    )

    # Feedback selection parameters (inspired by reference_code.py)
    ap.add_argument(
        "--feedback-top-k",
        type=int,
        default=3,
        help="Number of top-performing diverse specs to include in feedback context. Default: 3",
    )

    ap.add_argument(
        "--feedback-min-gap",
        type=float,
        default=0.005,
        help="Minimum error gap between selected feedback specs for diversity. Default: 0.005",
    )

    ap.add_argument(
        "--target-error",
        type=float,
        default=0.01,
        help="Target error threshold for early stopping. Default: 0.01",
    )

    ap.add_argument(
        "--no-improvement-patience",
        type=int,
        default=3,
        help="Stop if no improvement for this many iterations. Default: 3",
    )

    # JSON Schema option
    ap.add_argument(
        "--use-json-schema",
        type=bool,
        default=True,
        help="Include JSON Schema in the task prompt for structured output (default: True)",
    )

    # ========================================================================
    # Multi-dataset batch processing arguments
    # ========================================================================
    ap.add_argument(
        "--data-all-path",
        type=str,
        default=os.path.join(script_dir, "data_all"),
        help="Path to the data_all directory containing all datasets (default: data_all/).",
    )
    ap.add_argument(
        "--result-all-path",
        type=str,
        default=os.path.join(script_dir, "result_all"),
        help="Path to save all results (default: result_all/).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["all", "include", "exclude"],
        help="Dataset selection mode: 'all' (process all), 'include' (only specified), 'exclude' (all except specified).",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        nargs='*',
        default=[],
        help="List of dataset names to include or exclude (used with --mode include/exclude).",
    )
    ap.add_argument(
        "--categories",
        type=str,
        nargs='*',
        default=[],
        help="List of categories to process (e.g., ATVA FaMoS linear non_linear). Empty means all.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for batch processing (default: 4).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets that already have results in result_all/ (default: True).",
    )
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Force rerun even if results exist (overrides --skip-existing).",
    )

    args = ap.parse_args()

    # Validation depends on whether batch mode is enabled
    if args.mode is not None:
        # Batch mode: validate data_all_path
        if not os.path.exists(args.data_all_path):
            raise FileNotFoundError(f"data_all path {args.data_all_path} does not exist.")
    else:
        # Single dataset mode: validate input_data_path
        if not args.input_data_path:
            raise ValueError("You must provide a data path with --input-data-path.")
        if not os.path.exists(args.input_data_path):
            raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


# ============================================================================
# Single dataset processing function (core logic extracted from original main)
# ============================================================================

def process_single_dataset(
    dataset_info: DatasetInfo,
    output_dir: str,
    args
) -> DatasetResult:
    """
    Process a single dataset through the HA-Scientist loop.

    Args:
        dataset_info: Information about the dataset to process
        output_dir: Directory to save results for this dataset
        args: Parsed command-line arguments

    Returns:
        DatasetResult with processing outcome
    """
    start_time = time.time()
    input_data_path = dataset_info.path
    dataset_id = f"{dataset_info.category}/{dataset_info.name}"

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_id}")
    print(f"Data path: {input_data_path}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")

    try:
        # Auto-detect dimensions from data file
        num_variables, num_inputs = get_data_dimensions(input_data_path)

        # Create hyperparameters instance
        managed_agents = args.managed_agents_list if hasattr(args, 'managed_agents_list') and args.managed_agents_list else []
        hyperparameters = HAHyperparameters(
            manager_model=args.manager_model,
            manager_type=args.manager_type,
            managed_agents_count=len(managed_agents),
            managed_agents_list=managed_agents,
            managed_agents_model=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else "",
            tools_list=args.tools_list if args.tools_list else [],
            image_tool_model=args.image_tool_model,
            review_tool_model=args.review_tool_model,
            summarize_tool_model=args.summarize_tool_model,
            max_iterations=args.max_iterations,
            feedback_top_k=args.feedback_top_k,
            feedback_min_gap=args.feedback_min_gap,
            target_error=args.target_error,
            no_improvement_patience=args.no_improvement_patience,
            input_data_path=input_data_path,
            system_name=dataset_info.name,
            num_variables=num_variables,
            num_inputs=num_inputs,
            use_json_schema=args.use_json_schema,
        )

        # Create the agent
        managerAgent = create_agent(
            model_id=args.manager_model,
            input_data_path=input_data_path,
            tools_list=args.tools_list,
            image_tool_model=args.image_tool_model,
            review_tool_model=args.review_tool_model,
            summarize_tool_model=args.summarize_tool_model,
            managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
            managed_agents_list_model_id=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else None,
            manager_type=args.manager_type,
            tools_to_remove=args.tools_to_remove,
        )

        # Initialize results aggregator
        results_aggregator = ResultsAggregator(
            top_k=args.feedback_top_k,
            min_gap=args.feedback_min_gap
        )

        # HA-Scientist Iterative Loop
        for iteration in range(1, args.max_iterations + 1):
            print(f"\n[{dataset_id}] Iteration {iteration}/{args.max_iterations}")

            # Generate feedback context
            if iteration == 1:
                current_feedback = ""
            else:
                current_feedback = results_aggregator.get_top_k_feedback()
                latest_feedback = results_aggregator.get_latest_feedback()
                if latest_feedback and latest_feedback not in current_feedback:
                    current_feedback += f"\n\n## Most Recent Attempt (Iteration {iteration - 1}):\n{latest_feedback}"

            # Obtain task and images
            task, compressed_trace_images = obtain_task_and_images(
                input_data_path=input_data_path,
                system_name=dataset_info.name,
                num_variables=num_variables,
                num_inputs=num_inputs,
                initial_ha_spec=None,
                tools_list=args.tools_list,
                managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
                manager_type=args.manager_type,
                feedback=current_feedback,
                iteration=iteration,
                use_json_schema=args.use_json_schema
            )

            # Save task prompt
            task_filename = os.path.join(output_dir, 'task_prompts', f'iter_{iteration}_task.md')
            os.makedirs(os.path.dirname(task_filename), exist_ok=True)
            with open(task_filename, "w", encoding="utf-8") as f:
                f.write(task)

            # Run the agent
            try:
                result = managerAgent.run(task, images=compressed_trace_images)
            except Exception as e:
                print(f"[{dataset_id}] Agent execution failed: {e}")
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
                input_data_path,
                output_dir=os.path.join(output_dir, "evaluation_results", f"iter_{iteration}"),
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
                elif 'rmse' in metrics:
                    current_error = metrics['rmse']

            print(f"[{dataset_id}] Iteration {iteration} Error: {current_error}")

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
                print(f"[{dataset_id}] Early Stop: {stop_reason}")
                break

        # Save best result
        elapsed_time = time.time() - start_time
        best_result = results_aggregator.best_result

        if best_result and best_result.ha_specification:
            # Save best HA specification
            best_spec_path = os.path.join(output_dir, "best_ha_specification.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(best_spec_path, "w", encoding="utf-8") as f:
                json.dump(best_result.ha_specification, f, indent=2)

            # Save run summary
            run_summary = {
                "dataset": dataset_id,
                "success": True,
                "best_iteration": best_result.iteration,
                "best_error": results_aggregator.best_error,
                "total_iterations": len(results_aggregator.results),
                "elapsed_time_seconds": elapsed_time,
                "metrics": best_result.metrics
            }
            summary_path = os.path.join(output_dir, "run_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(run_summary, f, indent=2)

            return DatasetResult(
                dataset_info=dataset_info,
                success=True,
                metrics=best_result.metrics,
                ha_specification=best_result.ha_specification,
                elapsed_time=elapsed_time,
                best_iteration=best_result.iteration
            )
        else:
            return DatasetResult(
                dataset_info=dataset_info,
                success=False,
                metrics={},
                ha_specification=None,
                elapsed_time=elapsed_time,
                error_message="No successful HA specification generated"
            )

    except Exception as e:
        import traceback
        error_msg = f"Error processing {dataset_id}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return DatasetResult(
            dataset_info=dataset_info,
            success=False,
            metrics={},
            ha_specification=None,
            elapsed_time=time.time() - start_time,
            error_message=error_msg
        )


def run_batch_processing(
    datasets: List[DatasetInfo],
    result_all_path: str,
    args
) -> List[DatasetResult]:
    """
    Process multiple datasets in parallel.

    Args:
        datasets: List of datasets to process
        result_all_path: Base path for saving results
        args: Parsed command-line arguments

    Returns:
        List of DatasetResult for all processed datasets
    """
    results = []
    datasets_to_process = []

    # Filter out datasets that should be skipped
    for dataset_info in datasets:
        if should_skip_dataset(dataset_info, result_all_path, args.skip_existing, args.force_rerun):
            # Create a skipped result
            results.append(DatasetResult(
                dataset_info=dataset_info,
                success=True,
                metrics={},
                ha_specification=None,
                elapsed_time=0.0,
                error_message="Skipped (existing results)"
            ))
        else:
            datasets_to_process.append(dataset_info)

    if not datasets_to_process:
        print("No datasets to process (all skipped).")
        return results

    print(f"\n{'#'*60}")
    print(f"BATCH PROCESSING: {len(datasets_to_process)} datasets")
    print(f"Max workers: {args.max_workers}")
    print(f"{'#'*60}")

    # Process datasets in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_dataset = {}
        for dataset_info in datasets_to_process:
            output_dir = os.path.join(
                result_all_path,
                dataset_info.category,
                dataset_info.name
            )
            future = executor.submit(process_single_dataset, dataset_info, output_dir, args)
            future_to_dataset[future] = dataset_info

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_dataset):
            dataset_info = future_to_dataset[future]
            try:
                result = future.result()
                results.append(result)
                status = "SUCCESS" if result.success else "FAILED"
                print(f"[{status}] {dataset_info.category}/{dataset_info.name}")
            except Exception as e:
                print(f"[ERROR] {dataset_info.category}/{dataset_info.name}: {e}")
                results.append(DatasetResult(
                    dataset_info=dataset_info,
                    success=False,
                    metrics={},
                    ha_specification=None,
                    elapsed_time=0.0,
                    error_message=str(e)
                ))

    return results


def generate_summary_report(
    results: List[DatasetResult],
    result_all_path: str
) -> None:
    """
    Generate a summary report for all processed datasets.

    Args:
        results: List of DatasetResult from batch processing
        result_all_path: Path to save the summary report
    """
    successful = [r for r in results if r.success and r.ha_specification is not None]
    failed = [r for r in results if not r.success]
    skipped = [r for r in results if r.success and r.ha_specification is None]

    # Calculate statistics
    total_time = sum(r.elapsed_time for r in results)
    avg_time = total_time / len(results) if results else 0

    errors = [r.metrics.get('max_diff', float('inf')) for r in successful if 'max_diff' in r.metrics]
    avg_error = sum(errors) / len(errors) if errors else float('inf')

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_datasets": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_time_seconds": total_time,
            "avg_time_per_dataset_seconds": avg_time,
            "avg_max_diff_error": avg_error if avg_error < float('inf') else None
        },
        "datasets": {}
    }

    # Add per-dataset results
    for r in results:
        dataset_id = f"{r.dataset_info.category}/{r.dataset_info.name}"
        report["datasets"][dataset_id] = {
            "success": r.success,
            "elapsed_time_seconds": r.elapsed_time,
            "best_iteration": r.best_iteration,
            "metrics": r.metrics if r.metrics else None,
            "error_message": r.error_message if r.error_message else None
        }

    # Save report
    report_path = os.path.join(result_all_path, "summary_report.json")
    os.makedirs(result_all_path, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total datasets: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"Success rate: {len(successful)/len(results)*100:.1f}%" if results else "N/A")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per dataset: {avg_time:.1f}s")
    if avg_error < float('inf'):
        print(f"Avg max_diff error: {avg_error:.6f}")
    print(f"\nSummary report saved to: {report_path}")


def main():
    args = parse_args()

    # ========================================================================
    # Dispatch: Batch mode vs Single dataset mode
    # ========================================================================
    if args.mode is not None:
        # BATCH MODE: Process multiple datasets from data_all
        print(f"\n{'#'*60}")
        print("BATCH MODE: Processing multiple datasets")
        print(f"{'#'*60}")
        print(f"Data source: {args.data_all_path}")
        print(f"Results destination: {args.result_all_path}")
        print(f"Selection mode: {args.mode}")
        print(f"Categories filter: {args.categories if args.categories else 'ALL'}")
        print(f"Datasets filter: {args.datasets if args.datasets else 'N/A'}")
        print(f"Skip existing: {args.skip_existing}")
        print(f"Force rerun: {args.force_rerun}")
        print(f"Max workers: {args.max_workers}")

        # Discover all datasets
        all_datasets = discover_datasets(args.data_all_path)

        if not all_datasets:
            print("No datasets found. Exiting.")
            return

        # Filter datasets based on mode and selection
        datasets = filter_datasets(
            all_datasets,
            mode=args.mode,
            dataset_names=args.datasets,
            category_names=args.categories
        )

        print(f"\nSelected {len(datasets)} datasets for processing:")
        for d in datasets:
            print(f"  - {d.category}/{d.name}")

        # Run batch processing
        results = run_batch_processing(datasets, args.result_all_path, args)

        # Generate summary report
        generate_summary_report(results, args.result_all_path)

    else:
        # SINGLE DATASET MODE: Original behavior
        print(f"Running HA Learning Agent with model: {args.manager_model}, tools: {args.tools_list}, data path: {args.input_data_path}")

        # Create a DatasetInfo for the single dataset
        # Extract category and name from path
        path_parts = os.path.normpath(args.input_data_path).split(os.sep)
        dataset_name = path_parts[-1] if path_parts else "unknown"
        category = path_parts[-2] if len(path_parts) > 1 else "single"

        dataset_info = DatasetInfo(
            category=category,
            name=dataset_name,
            path=args.input_data_path
        )

        # Process single dataset
        result = process_single_dataset(
            dataset_info=dataset_info,
            output_dir="evaluation_results",
            args=args
        )

        # Print result
        if result.success:
            print(f"\nSingle dataset processing completed successfully.")
            print(f"Best iteration: {result.best_iteration}")
            print(f"Elapsed time: {result.elapsed_time:.1f}s")
            if result.metrics:
                print(f"Metrics: {result.metrics}")
        else:
            print(f"\nSingle dataset processing failed.")
            print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_llm_ha_beta.py --input-data-path utils/Dainarx_code/data_duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis summarize_hybrid_automaton_iterations validate_hybrid_automaton_specification
