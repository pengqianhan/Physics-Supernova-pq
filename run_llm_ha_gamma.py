import os
import sys
import json
import secrets
import shutil
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from utils.prompt import ha_spec_docs
import base64
from litellm import completion

def generate_run_id() -> str:
    """
    Generate a unique run ID for tracking experiment artifacts.
    Format: YYYYMMDD_HHMMSS_<4-char-hex>
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = secrets.token_hex(2)  # 4 hex characters
    return f"{timestamp}_{suffix}"
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# === Agent Monitoring (Phoenix - Free Local Solution) ===
PHOENIX_AVAILABLE = False
PHOENIX_PROJECT_NAME = None  # Global variable to store current project name
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    # Use timestamped project name to separate runs
    PHOENIX_PROJECT_NAME = f"ha_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    register(project_name=PHOENIX_PROJECT_NAME)
    SmolagentsInstrumentor().instrument()
    PHOENIX_AVAILABLE = True
    print(f"[Telemetry] Phoenix monitoring enabled (project: {PHOENIX_PROJECT_NAME}) - visit http://localhost:6006")
except ImportError:
    print("[Telemetry] Phoenix not installed. Install: pip install arize-phoenix openinference-instrumentation-smolagents")
# =============================================


import argparse

# for type hints
from typing import List, Dict, Tuple, Optional, Union
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
from utils.Dainarx_code.HA_evaluation import HAEvaluator
from utils.ha_spec_validator import preprocess_ha_for_evaluation
from utils.ha_structured_output import preprocess_ha_for_evaluation_v2, convert_agent_result_to_ha
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


def generate_dynamic_ha_template(num_variables: int, num_inputs: int) -> str:
    """
    Generate a dynamic HA specification template with pre-filled var and input fields.
    
    Args:
        num_variables: Number of state variables
        num_inputs: Number of input variables
        
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
        eq_placeholder = "x1[2] = x1[1] + x1[0]"
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
        "dt":,
        "total_time":,
        "order": {dim},
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

# Shared authorized imports for both manager and managed agents
AUTHORIZED_IMPORTS_LIST = [
    "os", "sys", "time", "argparse", "pathlib",
    "matplotlib.pyplot", "matplotlib", "pandas", "json",
    # numpy and all common submodules
    "numpy", "numpy.linalg", "numpy.fft", "numpy.random",
    "numpy.polynomial", "numpy.ma", "numpy.lib","numpy.diff",
    # scipy and all common submodules
    "scipy", "scipy.linalg", "scipy.optimize", "scipy.interpolate",
    "scipy.integrate", "scipy.stats", "scipy.signal", "scipy.fft",
    "scipy.sparse", "scipy.ndimage", "scipy.special",
    # optimization and system identification
    "pysindy", "gradient_free_optimizers", "gradient_free_optimizers.BayesianOptimizer",
]


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
        # max_completion_tokens=24576,
        num_retries=3,
        timeout=1200,
        thinking_level = "high" # high, low
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
        add_base_tools=True 
    )

    if kwargs["manager_type"] == "CodeAgent":
        manager_agent_kwargs["additional_authorized_imports"] = AUTHORIZED_IMPORTS_LIST
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

    return managerAgent


def get_managed_agents_list(managed_agents_list: List[str] = None,
                            managed_agents_list_model_id: str = None,
                            input_data_path: str = None,
                            markdown_content: MarkdownMessage = None,
                            train_num: int = 3) -> List[MultiStepAgent]:
    if managed_agents_list is None:
        return []

    # Load trace data to get NPZ paths if not provided
    if markdown_content is None:
        markdown_content = load_trace_data_from_filepath(input_data_path, train_num=train_num)

    # Get NPZ file paths from markdown_content
    # npz_paths is a dict like {"<image_0>": "/path/to/sample_0.npz", ...}
    npz_paths_list = list(markdown_content.npz_paths.values()) if markdown_content.npz_paths else []
    npz_placeholders = list(markdown_content.npz_paths.keys()) 

    # Fallback to default path if no npz files found
    if not npz_paths_list:
        npz_paths_list = [os.path.join(input_data_path, "sample_0.npz")]

    managed_agents = []
    model = LiteLLMModel(
            model_id=managed_agents_list_model_id,
            api_key=os.environ.get("GEMINI_API_KEY"),
            # max_completion_tokens=24576,
            num_retries=3,
            timeout=1200,
            thinking_level = "high" # high, low
        )
    managed_agent_kwargs = dict(
        model=model,
        tools=[],
        max_steps=80,
        verbosity_level=2,
        add_base_tools=True,
        additional_authorized_imports=AUTHORIZED_IMPORTS_LIST,
    )
    npz_files_description = "\n".join([f" {placeholder} - `{path}` - `DATA_FILE_PATHS[{i}]`" for i, (placeholder, path) in enumerate(zip(npz_placeholders, npz_paths_list))])
    for agent_name in managed_agents_list:

        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can assist with code-related tasks."""
        common_instruction = """
### Quick Access via State Variables

- `DATA_FILE_PATHS`: List of all available NPZ file paths

### Example Usage

```python
import numpy as np

# Load a the <npz_0>
data = np.load(DATA_FILE_PATHS[0])

# Or use the primary file"""
        # use_e2b = bool(os.environ.get("E2B_API_KEY"))
        use_e2b = False
        if use_e2b:
            print("使用 E2B 云沙盒执行器，正在上传数据文件...")
            # 上传所有文件到 E2B 沙盒
            sandbox_file_paths = []
            for i, npz_path in enumerate(npz_paths_list):
                with open(npz_path, "rb") as f:
                    file_content = f.read()
                sandbox_file_path = f"/tmp/sample_{i}.npz"
                managed_agent.python_executor.sandbox.files.write(sandbox_file_path, file_content)
                sandbox_file_paths.append(sandbox_file_path)
                print(f"✓ 文件已上传到 E2B 沙盒: {sandbox_file_path}")
            managed_agent_kwargs["executor_type"] = "e2b"
            managed_agent_kwargs["name"] = agent_name
            managed_agent_kwargs["description"] = managed_agent_description
            managed_agent_kwargs["instructions"] = managed_agent_instruction
            managed_agent = CodeAgent(**managed_agent_kwargs)
            # 注入所有文件路径到 agent 状态
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = sandbox_file_paths
            managed_agent.python_executor.state["DATA_FILE_PATH"] = sandbox_file_paths[0] if sandbox_file_paths else ""
        else:
            # managed agent description with all available files
            
            managed_agent_instruction = f"""## Available NPZ Files

You have access to the following NPZ data files:

{npz_files_description}""" + common_instruction
            # 本地执行器：将所有数据文件路径注入到 agent 的状态中
            managed_agent_kwargs["name"] = agent_name
            managed_agent_kwargs["executor_type"] = "local"
            managed_agent_kwargs["description"] = managed_agent_description
            managed_agent_kwargs["instructions"] = managed_agent_instruction
            managed_agent = CodeAgent(**managed_agent_kwargs)
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = npz_paths_list
            managed_agent.python_executor.state["DATA_FILE_PATH"] = npz_paths_list[0] if npz_paths_list else ""
        managed_agents.append(managed_agent)
        # save the instruction and description to file
        # with open(f"managed_agent_{agent_name}_instruction.md", "w") as f:
        #     f.write(managed_agent_instruction)

        # managed_agent_prompt = managed_agent.prompt_templates["system_prompt"]
        # with open(f"managed_agent_{agent_name}_prompt.md", "w") as f:
        #     f.write(managed_agent_prompt)

    return managed_agents


# create the agent
def create_agent(model_id: str = "gemini/gemini-flash-lite-latest",
                input_data_path: str = None,
                tools_list: List[str] = [],
                managed_agents_list: List[str] = None,
                managed_agents_list_model_id: str = None,
                train_num: int = 3,
                **kwargs) -> ToolCallingAgent | CodeAgent:
    '''
    Create a Hybrid Automaton learning agent.

    Args:
        model_id: Model ID for the main agent
        input_data_path: Path to the trace data directory
        tools_list: List of tool names to use
        managed_agents_list: List of managed agent names
        managed_agents_list_model_id: Model ID for managed agents
        train_num: Number of training samples to load
        **kwargs: Additional arguments (manager_type, image_tool_model, review_tool_model, etc.)

    Returns:
        Configured ToolCallingAgent or CodeAgent
    '''
    # Load trace data with high res images
    markdown_content = load_trace_data_from_filepath(input_data_path, train_num=train_num)

    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]
    haAgent = _create_HA_agent(
        Tools_list=ToolsList,
        markdown_content=markdown_content,
        model_id=model_id,
        managed_agents_list=get_managed_agents_list(
            managed_agents_list,
            managed_agents_list_model_id,
            input_data_path,
            markdown_content=markdown_content,  # Pass markdown_content to reuse loaded npz paths
            train_num=train_num
        ),
        **kwargs
    )
    return haAgent


# obtain task string and images for the agent to run
def obtain_task_and_images(input_data_path: str = None,
                           num_variables: int = 1,
                           num_inputs: int = 1,
                           initial_ha_spec: str = None,
                           tools_list: List[str] = [],
                           managed_agents_list: List[str] = None,
                           manager_type: str = "CodeAgent",
                           feedback: str = None, # Added feedback parameter
                           iteration: int = 1,
                           use_json_schema: bool = True,
                           train_num: int = 3) -> tuple[str, list]:
    '''
    Generate task prompt and compressed images for HA learning agent.

    Args:
        input_data_path: Path to the trace data directory
        num_variables: Number of state variables
        num_inputs: Number of input variables
        initial_ha_spec: Initial HA specification dictionary (optional)
        tools_list: List of tool names
        managed_agents_list: List of managed agent names
        manager_type: Type of manager agent
        feedback: Feedback string from previous iteration (optional)
        iteration: Current iteration number (1-indexed)
        use_json_schema: If True, include JSON Schema in the prompt for structured output
        train_num: Number of training samples to load

    Returns:
        Tuple of (task prompt string, list of compressed images)
    '''
    print(f"Generating task for iteration {iteration}, tools_list: {tools_list}")

    # Load trace data with high res images
    markdown_content = load_trace_data_from_filepath(input_data_path, train_num=train_num)
    image_paths = markdown_content.image_paths## list(image_paths.keys())
    image_placeholders = list(image_paths.keys())
    npz_paths = markdown_content.npz_paths## list(npz_paths.keys())
    npz_placeholders = list(npz_paths.keys())

    # create the manager agent
    ToolsList = [TOOLNAME2TOOL[x] for x in tools_list]

    # Get task and images (to parse into agents) from the markdown content
    trace_data_text = markdown_to_plaintext(markdown_content)
    compressed_trace_images = markdown_images_compress(markdown_content, max_short_side_pixels=1080)

    # Base task prompt - Professional system identification framing
    task = f"""# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data.

## Problem Context
A Hybrid Automaton models a system with:
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each mode)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)


## Metrics (lower is better)
- `Max Difference < 0.002`: good fit
- `Mean Difference < 0.001`: good fit
"""

    task += "\n## Tool and Sub-Agents Resources:\n"
    # Add tool-specific prompts (unified style: init empty -> conditionally set -> unconditionally append)
    HA_IMAGE_TOOL_PROMPT = ""
    if HybridAutomatonImageTool in ToolsList:
        HA_IMAGE_TOOL_PROMPT = "\nYou MUST use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system."

    REVIEW_TOOL_PROMPT = ""
    if ReviewRequestTool_ha in ToolsList:
        REVIEW_TOOL_PROMPT = """You MUST call the `hybrid_automaton_review_expert` tool at least once before submitting your final answer to ensure specification correctness and completeness."""

    VALIDATE_TOOL_PROMPT = ""
    if ValidateHASpecTool in ToolsList:
        VALIDATE_TOOL_PROMPT = """Before submitting your final answer, you MUST call the `validate_hybrid_automaton_specification` tool to check for syntax and semantic errors. If validation returns a FIXED specification, use the corrected version in your final answer!"""

    task += HA_IMAGE_TOOL_PROMPT
    task += REVIEW_TOOL_PROMPT
    task += VALIDATE_TOOL_PROMPT
    # Add managed agents prompt
    if managed_agents_list and len(managed_agents_list) > 0:
        MANAGE_AGENT_PROMPT = f"""
\nYou have access to managed Code Agent: `{managed_agents_list}` to analyze the npz data files."""
        task += MANAGE_AGENT_PROMPT


    # Generate dynamic HA template with correct var and input fields pre-filled
    dynamic_ha_template = generate_dynamic_ha_template(num_variables, num_inputs)
    
    # Generate variable and input names for display
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else "(none)"

    # Add HA specification format documentation and initial spec
    # Use JSON Schema-based documentation for more precise output specification
    # We use the explicit schema and examples to ensure consistency with the improved task prompt
    

    task += f"""
{ha_spec_docs}

## CRITICAL: Variable Count is PRE-DEFINED
- The `var` and `input` fields in  Initial Hybrid Automaton Specification (v0) are **already correctly set** based on the ground truth data. **DO NOT** add or remove variables!
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- ODE notation:
   - `x[0]` = variable value
   - `x[1]` = first derivative (dx/dt)
   - `x[2]` = second derivative (d²x/dt²)
   - Left side = highest derivative (e.g., `x1[2] = ...` for order=2)
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)

## IMPORTANT: Python Boolean Syntax
When constructing the HA specification dictionary in Python code:
- Use `True`/`False` (Python syntax), **NOT** `true`/`false` (JSON syntax)
- Example: `"need_reset": True` not `"need_reset": true`
- Example: `"self_loop": False` not `"self_loop": false`

## Initial Hybrid Automaton Specification (v0)

```json
{dynamic_ha_template}
```

## Your Task
Generate an improved HA specification that better matches the observed trajectory data.
- **Keep `var: "{var_names}"` and `input: "{input_names if num_inputs > 0 else ''}"` exactly as shown!**
- Make sure the HA specification is valid and complete according to the JSON Schema.
- Refine the HA specification to improve trajectory matching and reduce `Max Difference`, `Mean Difference`.

## Available Data
- **Trace visualizations**: {image_placeholders}, use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.
- **Raw data files**: If you want to use the npz data to analyze the system, you MUST use the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.

### NPZ Data Format (for reference - use via `data_analysis_expert` agent)
Each NPZ file contains:
- `state`: numpy array, shape `(num_variables, num_steps)` - state trajectories
  - Access: `x1 = data['state'][0, :]`, `x2 = data['state'][1, :]`
- `input`: numpy array, shape `(num_inputs, num_steps)` - input signals (if applicable)
  - Access: `u1 = data['input'][0, :]`

**WARNING**: Do NOT use placeholder names like `<npz_0>` as file paths! Use the `data_analysis_expert` agent which has access to the actual file paths.

### Detecting Mode Transitions via Numerical Analysis
For higher-order systems (order >= 2), position trajectories may appear smooth even when mode transitions occur, because resets often affect **derivatives** (velocity, acceleration) rather than position directly. Use the `data_analysis_expert` agent to:
1. **Compute numerical derivatives**: `velocity = np.diff(state, axis=1) / dt` and `acceleration = np.diff(velocity, axis=1) / dt`
2. **Detect discontinuities**: sudden jumps in velocity or acceleration indicate potential mode transition points and resets
3. **Segment-wise analysis**: once candidate transition points are identified, analyze each segment's dynamics separately to infer mode-specific ODEs and guard conditions

"""

    # Add feedback from previous iteration if available
    if feedback:
        task += f""""\n## Previously Explored HA Specifications with Feedback:\nUse these as inspiration to guide your next refinement.\nUse the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section.\n If you want to check the fit performance of the HA specification, you can use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section. For example, `hybrid_automaton_image_analysis(image_ref='<iter_image_2_0>', question='How is the fit performance of the HA specification?')` means to analyze the '2nd iteration, 0th file' comparison plot.`.
{feedback}"""

    return task, compressed_trace_images


def compute_aggregated_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Compute average metrics across multiple evaluations.

    Args:
        metrics_list: List of metrics dictionaries from individual evaluations

    Returns:
        Dictionary containing averaged metrics and per-file details
    """
    if not metrics_list:
        return {}
    avg = {}
    for key in ['max_diff', 'mean_diff']:
        vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if vals:
            avg[key] = sum(vals) / len(vals)
    return {**avg, 'num_evaluations': len(metrics_list), 'per_file_metrics': metrics_list}


def gen_summary(
    metrics_dict: Dict,
    ha_specification: Dict,
    plot_paths: Union[str, List[str]],
    model_id: str = "gemini/gemini-3-flash-preview"
) -> str:
    """
    Generate a comprehensive summary of evaluation results using LLM with vision.

    Analyzes the metrics, HA specification, and evaluation plot(s) to provide
    actionable feedback for improving the hybrid automaton model.

    Args:
        metrics_dict: Dictionary containing evaluation metrics (mean_diff, max_diff, etc.)
        ha_specification: The HA specification dictionary being evaluated
        plot_paths: Path(s) to overlay plot image(s) - can be a single string or list of strings
        model_id: LiteLLM model ID (default: gemini/gemini-3-flash-preview)

    Returns:
        LLM-generated summary with analysis and improvement suggestions
    """

    # Fallback if no metrics
    if not metrics_dict:
        return "No metrics available for analysis."

    # Normalize plot_paths to a list
    if isinstance(plot_paths, str):
        plot_paths_list = [plot_paths]
    else:
        plot_paths_list = plot_paths if plot_paths else []

    # Read and encode all plot images
    image_contents = []
    for idx, plot_path in enumerate(plot_paths_list):
        if plot_path and os.path.isfile(plot_path):
            try:
                with open(plot_path, 'rb') as f:
                    image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # Determine mime type from extension
                ext = os.path.splitext(plot_path)[1].lower()
                mime_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                })
            except Exception as e:
                print(f"[gen_summary] Warning: Could not read plot image {idx}: {e}")

    # Format metrics for prompt (handle aggregated metrics)
    metrics_lines = []
    for k, v in metrics_dict.items():
        if v is None:
            continue
        if k == 'per_file_metrics':
            # Skip per-file details in text summary, they're in the images
            continue
        if isinstance(v, float):
            metrics_lines.append(f"- {k}: {v:.6f}")
        else:
            metrics_lines.append(f"- {k}: {v}")
    metrics_text = "\n".join(metrics_lines)

    # Format HA spec summary (truncate if too long)
    ha_spec_str = json.dumps(ha_specification, indent=2)
    if len(ha_spec_str) > 2000:
        ha_spec_str = ha_spec_str[:2000] + "\n... (truncated)"

    # Build the analysis prompt
    num_plots = len(image_contents)
    plot_description = f"{num_plots} trajectory comparison plot(s)" if num_plots > 0 else "trajectory comparison plot (not available)"

    system_prompt = """You are an expert in hybrid automaton system identification and control systems.
Analyze the evaluation results and provide a concise, actionable summary.

Focus on:
1. Overall fit quality based on metrics
2. Visual patterns in the trajectory comparison (if image provided)
3. Specific issues: amplitude errors, phase lag, mode switch timing, divergence
4. Concrete suggestions for improving the HA specification

Keep the summary under 200 words. Be direct and technical."""

    user_prompt = f"""## Evaluation Metrics
{metrics_text}

## HA Specification
```json
{ha_spec_str}
```

## Task
Analyze the evaluation results and the {plot_description} (ground truth vs simulated).
Identify the main sources of error and suggest specific improvements to the HA JSON specification including mode and edge."""

    # Build message content - add all images first, then text
    content = []
    for img_content in image_contents:
        content.append(img_content)
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    # Call LLM with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_id,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
                api_key=os.environ.get("GEMINI_API_KEY")
            )
            summary = response.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception as e:
            print(f"[gen_summary] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)

    return "LLM summary generation failed after all retries."


def build_artifact_manifest(
    run_id: str,
    iteration: int,
    metrics: Dict,
    plot_placeholders: Union[str, List[str]],
    plot_summary: str,
    per_file_results: List[Dict] = None
) -> Dict:
    """
    Build the artifact manifest structure for feedback.

    Args:
        run_id: Unique identifier for this experiment run
        iteration: Current iteration number
        metrics: Evaluation metrics dictionary (may be aggregated if multiple files)
        plot_placeholders: Placeholder reference(s) for overlay plot(s)
        plot_summary: Text summary for fallback
        per_file_results: Optional list of per-file evaluation results

    Returns:
        Dictionary containing the artifact manifest
    """
    # Normalize plot_placeholders to list
    if isinstance(plot_placeholders, str):
        placeholders_list = [plot_placeholders]
    else:
        placeholders_list = plot_placeholders if plot_placeholders else []

    manifest = {
        "Evaluation Metrics (Averaged)": {
            "Max Difference": metrics.get('max_diff', None),
            "Mean Difference": metrics.get('mean_diff', None),
            "Num Ground Truth Files": metrics.get('num_evaluations', 1),
        },
        "Evaluation Plot": {
            "Placeholders": placeholders_list,
            "Summary": plot_summary,
        }
    }

    # Add per-file results if available
    if per_file_results:
        manifest["Per-File Results"] = per_file_results

    return manifest


def evaluate_ha_specification_with_feedback(
    agent_result,
    input_data_path: str,
    output_dir: str = None,
    hyperparameters: HAHyperparameters = None,
    iteration: int = 1,
    use_structured_output: bool = True,
    structured_output_model: str = "gemini/gemini-3-flash-preview",
    run_id: str = None,
    summary_model: str = "gemini/gemini-3-flash-preview",
    image_tool: HybridAutomatonImageTool = None,
    eval_train_num: int = 1
) -> Tuple[bool, Dict, str, Optional[Dict]]:
    """
    Evaluate the generated Hybrid Automaton specification and return feedback for the agent.

    Args:
        agent_result: Result from the agent.run() call
        input_data_path: Path to the directory containing test .npz files
        output_dir: Directory to save evaluation results
        hyperparameters: HAHyperparameters instance containing experiment configuration
        iteration: Current iteration number (for logging)
        use_structured_output: 是否启用两阶段结构化输出转换 (default: True)
        structured_output_model: 结构化输出使用的模型 ID
        run_id: Unique run identifier for artifact tracking
        summary_model: Model ID for LLM-based summary generation
        image_tool: HybridAutomatonImageTool instance for registering evaluation plots
        eval_train_num: Number of ground truth files to evaluate against (default: 1)

    Returns:
        Tuple of (success_bool, metrics_dict, feedback_string, ha_specification_dict)
        The ha_specification_dict is the extracted/validated HA spec (None if extraction failed)
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Testing the generated Hybrid Automaton specification")
    print("=" * 80)

    # 两阶段架构: 先尝试传统提取，失败则使用 LLM 结构化输出
    print("\n--- HA Specification Extraction & Validation ---")
    if use_structured_output:
        # print(f"Using structured output converter")
        ha_specification, is_valid, validation_message = preprocess_ha_for_evaluation_v2(
            agent_result
        )
    else:
        # print("Using traditional extraction only")
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

    # Find test data files
    # Ground truth files are in a folder with the same name but with "_g" suffix
    # e.g., if input_data_path is "data_all/ATVA/ball", ground truth is in "data_all/ATVA/ball_g"
    ground_truth_path = input_data_path.rstrip('/') + '_g'

    if os.path.isdir(ground_truth_path):
        test_data_files = [f for f in os.listdir(ground_truth_path) if f.startswith('ground_truth') and f.endswith('.npz')]
        test_data_base_path = ground_truth_path
    else:
        # Fallback: look in the original input_data_path
        test_data_files = [f for f in os.listdir(input_data_path) if f.startswith('ground_truth') and f.endswith('.npz')]
        test_data_base_path = input_data_path

    if not test_data_files:
        return False, {}, f"No .npz test data files found in {ground_truth_path} or {input_data_path}", ha_specification

    # Sort files by numeric index (ground_truth_0.npz, ground_truth_1.npz, ...)
    def extract_index(filename):
        # Extract number from filename like "ground_truth_0.npz" -> 0
        import re
        match = re.search(r'ground_truth_(\d+)\.npz', filename)
        return int(match.group(1)) if match else float('inf')

    test_data_files.sort(key=extract_index)

    # Limit to eval_train_num files
    test_data_files = test_data_files[:eval_train_num]
    print(f"Evaluating against {len(test_data_files)} ground truth file(s): {test_data_files}")

    # Check dimensions using the first file
    first_npz_path = os.path.join(test_data_base_path, test_data_files[0])
    gt_data = np.load(first_npz_path, allow_pickle=True)
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
        # Evaluate against each ground truth file
        all_metrics = []
        all_overlay_paths = []
        all_plot_placeholders = []
        per_file_results = []
        all_metrics_texts = []

        for file_idx, test_file in enumerate(test_data_files):
            npz_file_path = os.path.join(test_data_base_path, test_file)
            print(f"\n--- Evaluating against {test_file} ({file_idx + 1}/{len(test_data_files)}) ---")

            # Create evaluator for this file
            evaluator = HAEvaluator(
                ha_dict=ha_specification,
                npz_file_path=npz_file_path,
                dt=ha_specification['config'].get('dt', 0.001),
                total_time=ha_specification['config'].get('total_time', 10.0)
            )

            # Run evaluation - save as overlay_0.png, overlay_1.png, etc.
            overlay_filename = f'overlay_{file_idx}.png'
            overlay_path = os.path.abspath(os.path.join(output_dir, overlay_filename))
            metrics_text, _ = evaluator(
                plot_mode='overlay',
                save_path=overlay_path,
                print_metrics=True
            )
            file_metrics = evaluator.metrics

            all_metrics.append(file_metrics)
            all_overlay_paths.append(overlay_path)
            all_metrics_texts.append(metrics_text)

            # Create placeholder for this file's plot
            plot_placeholder = f"<iter_image_{iteration}_{file_idx}>"
            all_plot_placeholders.append(plot_placeholder)

            # Register the overlay image with the image tool
            if image_tool is not None:
                # Use a unique index combining iteration and file index
                unique_idx = iteration * 100 + file_idx  # e.g., iteration 1, file 0 -> 100
                success, error_msg = image_tool.register_iteration_image(unique_idx, overlay_path)
                if not success:
                    print(f"[Warning] Failed to register overlay image {file_idx}: {error_msg}")

            # Build per-file result entry
            per_file_results.append({
                "ground_truth_index": file_idx,
                "ground_truth_file": test_file,
                "plot_placeholder": plot_placeholder,
                "max_diff": file_metrics.get('max_diff'),
                "mean_diff": file_metrics.get('mean_diff')
            })

        # Compute aggregated metrics
        aggregated_metrics = compute_aggregated_metrics(all_metrics)
        print(f"\n--- Aggregated Metrics (over {len(all_metrics)} files) ---")
        print(f"  Mean Max Diff: {aggregated_metrics.get('max_diff', 'N/A'):.6f}" if aggregated_metrics.get('max_diff') is not None else "  Mean Max Diff: N/A")
        print(f"  Mean Mean Diff: {aggregated_metrics.get('mean_diff', 'N/A'):.6f}" if aggregated_metrics.get('mean_diff') is not None else "  Mean Mean Diff: N/A")

        # Save metrics and hyperparameters to metrics.txt
        metrics_file = os.path.join(output_dir, 'metrics.txt')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Run ID: {run_id if run_id else 'N/A'}\n")
            f.write(f"Number of Ground Truth Files: {len(test_data_files)}\n\n")

            # Write aggregated metrics
            f.write("Aggregated Metrics:\n")
            f.write(f"  Mean Max Diff: {aggregated_metrics.get('max_diff', 'N/A')}\n")
            f.write(f"  Mean Mean Diff: {aggregated_metrics.get('mean_diff', 'N/A')}\n\n")

            # Write per-file metrics
            f.write("Per-File Metrics:\n")
            for idx, (test_file, metrics_text) in enumerate(zip(test_data_files, all_metrics_texts)):
                f.write(f"\n  [{idx}] {test_file}:\n")
                f.write(f"    {metrics_text}\n")

            f.write(f"\nHA Specification:\n{json.dumps(ha_specification, indent=2)}\n\n")

            # Write hyperparameters if provided
            if hyperparameters is not None:
                f.write(hyperparameters.to_string())
                f.write("\n\n")

        # Generate plot summary using LLM analysis of metrics, HA spec, and all plot images
        plot_summary = gen_summary(aggregated_metrics, ha_specification, all_overlay_paths, model_id=summary_model)

        # Build artifact manifest with all placeholders and per-file results
        artifact_manifest = build_artifact_manifest(
            run_id=run_id if run_id else "unknown",
            iteration=iteration,
            metrics=aggregated_metrics,
            plot_placeholders=all_plot_placeholders,
            plot_summary=plot_summary,
            per_file_results=per_file_results
        )

        # Save artifact manifest to disk (for debugging/external tools)
        manifest_file = os.path.join(output_dir, 'artifacts.json')
        with open(manifest_file, 'w') as f:
            json.dump(artifact_manifest, f, indent=2)

        # Construct feedback string with HA spec, metrics, and artifact manifest
        feedback = f"The {iteration}th attempt result:\n"
        feedback += f"  1. HA JSON Specification:\n```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        feedback += f"  2. Evaluation Feedback:\n{json.dumps(artifact_manifest, indent=2)}\n"

        return True, aggregated_metrics, feedback, ha_specification

    except Exception as e:
        import traceback
        traceback.print_exc()
        feedback = f"```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        # feedback = f"```json\n{agent_result}\n```\n"
        feedback += f"Evaluation Results:\nThe HA specification is not valid. Please try to generate a valid specification. Here is the error message: {str(e)}"
        return False, {}, feedback, ha_specification


def parse_args():

    ap = argparse.ArgumentParser(description="Run the HA Learning Agent with specified tools and model.")
    ap.add_argument(
        "--input-data-path",
        type=str,
        default='data_all/ATVA/ball',
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
        default=["hybrid_automaton_image_analysis"],
        choices=["hybrid_automaton_image_analysis", "hybrid_automaton_review_expert", "summarize_hybrid_automaton_iterations", "validate_hybrid_automaton_specification"],
        help="List of tool names to use in the agent.",
    )

    # LLM model ids for the tools
    ap.add_argument(
        "--image-tool-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID to use for the image analysis tool (Gemini API format).",
    )
    ap.add_argument(
        "--review-tool-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID to use for the review tool (Gemini API format).",
    )
    ap.add_argument(
        "--summarize-tool-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID to use for the summarize iterations tool (Gemini API format).",
    )

    # Model for gen_summary (LLM-based evaluation analysis)
    ap.add_argument(
        "--summary-model",
        type=str,
        default="gemini/gemini-flash-lite-latest",
        help="Model ID for gen_summary LLM-based evaluation analysis (default: gemini/gemini-flash-lite-latest).",
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
        "--target-error",
        type=float,
        default=0.001,
        help="Target error threshold for early stopping. Default: 0.001",
    )

    ap.add_argument(
        "--no-improvement-patience",
        type=int,
        default=10,
        help="Stop if no improvement for this many iterations. Default: 10",
    )

    # JSON Schema option
    ap.add_argument(
        "--use-json-schema",
        type=bool,
        default=True,
        help="Include JSON Schema in the task prompt for structured output (default: True)",
    )

    # Two-stage structured output options
    ap.add_argument(
        "--use-structured-output",
        type=bool,
        default=True,
        help="Enable two-stage structured output conversion (default: True). "
             "Stage 1: Traditional extraction. Stage 2: LLM structured output if Stage 1 fails.",
    )
    ap.add_argument(
        "--structured-output-model",
        type=str,
        default="gemini/gemini-3-flash-preview",
        help="Model ID for structured output conversion (default: gemini/gemini-3-flash-preview)",
    )

    # Training data count
    ap.add_argument(
        "--train-num",
        type=int,
        default=3,
        help="Number of training samples to load from trace data (default: 3)",
    )

    args = ap.parse_args()

    if not args.input_data_path:
        raise ValueError("You must provide a data path with --input-data-path.")
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


def main():
    args = parse_args()
    # Generate unique run ID for this experiment session
    # This ID persists across all iterations and enables stable artifact referencing
    run_id = generate_run_id()

    # Auto-detect dimensions from data file (overrides command-line args if provided)
    num_variables, num_inputs = get_data_dimensions(args.input_data_path)

    # Create hyperparameters instance to track experiment configuration
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
        target_error=args.target_error,
        no_improvement_patience=args.no_improvement_patience,
        input_data_path=args.input_data_path,
        num_variables=num_variables,
        num_inputs=num_inputs,
        use_json_schema=args.use_json_schema,
        use_structured_output=args.use_structured_output,
        structured_output_model=args.structured_output_model,
    )

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
        train_num=args.train_num,
        manager_type=args.manager_type,
    )

    # ========================================================================
    # HA-Scientist Iterative Loop (Enhanced with ResultsAggregator)
    # Inspired by SR-Scientist\inference\infer\inference.py's multi-turn adaptive loop pattern
    # ========================================================================

    # Initialize results aggregator for intelligent feedback selection
    results_aggregator = ResultsAggregator(
        top_k=args.feedback_top_k
    )

    # Extract relative path from input_data_path for output directory structure
    # e.g., "data_all/ATVA/ball" -> "ATVA/ball"
    if args.input_data_path.startswith("data_all/"):
        relative_data_path = args.input_data_path[len("data_all/"):]
    elif args.input_data_path.startswith("data_all"):
        relative_data_path = args.input_data_path[len("data_all"):].lstrip("/")
    else:
        relative_data_path = os.path.basename(args.input_data_path)

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*40}")
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print(f"{'#'*40}")

        # Generate feedback context using intelligent top-k selection
        # For first iteration, no feedback available
        if iteration == 1:
            current_feedback = ""
        else:
            # Use top-k diverse feedback instead of raw concatenation
            current_feedback = results_aggregator.get_top_k_feedback()

        # Obtain task and images with feedback from previous iteration
        task, compressed_trace_images = obtain_task_and_images(
            input_data_path=args.input_data_path,
            num_variables=num_variables,
            num_inputs=num_inputs,
            initial_ha_spec=None,
            tools_list=args.tools_list,
            managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
            manager_type=args.manager_type,
            feedback=current_feedback,
            iteration=iteration,
            use_json_schema=args.use_json_schema,
            train_num=args.train_num
        )

        # save the task to a file and set up output directory
        task_filename = os.path.join(os.path.dirname(__file__), 'task_prompts', f'iter_{iteration}_task.md')
        os.makedirs(os.path.dirname(task_filename), exist_ok=True)
        with open(task_filename, "w", encoding="utf-8") as f:
            f.write(task)

        # Run the agent with task and compressed images
        try:
            result = managerAgent.run(task, images=compressed_trace_images)
        except Exception as e:
            # Create failed iteration result
            failed_result = IterationResult(
                iteration=iteration,
                ha_specification=None,
                metrics={},
                feedback=f"Agent execution failed to get result: {str(e)}",
                success=False,
                error_value=float('inf')
            )
            results_aggregator.add_result(failed_result)
            continue

        # Evaluate the generated HA specification
        # Use <relative_data_path>/runs/<run_id>/iter_<N> structure for stable artifact referencing
        eval_output_dir = os.path.join("evaluation_results", relative_data_path, "runs", run_id, f"iter_{iteration}")

        # Get the image tool from the agent for registering evaluation plots
        image_tool = managerAgent.tools.get('hybrid_automaton_image_analysis', None)

        success, metrics, feedback_str, ha_spec = evaluate_ha_specification_with_feedback(
            result,
            args.input_data_path,
            output_dir=eval_output_dir,
            hyperparameters=hyperparameters,
            iteration=iteration,
            use_structured_output=args.use_structured_output,
            structured_output_model=args.structured_output_model,
            run_id=run_id,
            summary_model=args.summary_model,
            image_tool=image_tool,
            eval_train_num=args.train_num
        )

        # Extract error value from metrics
        current_error = float('inf')
        if success and isinstance(metrics, dict) and 'mean_diff' in metrics:
            current_error = metrics['mean_diff']
            print(f"Iteration {iteration} mean_diff: {current_error:.6f}, max_diff: {metrics.get('max_diff', float('inf')):.6f}")
        else:
            print(f"Iteration {iteration} FAILED - evaluation did not produce valid metrics")

        # Create and store iteration result
        iter_result = IterationResult(
            iteration=iteration,
            ha_specification=ha_spec,
            metrics=metrics if isinstance(metrics, dict) else {},
            feedback=feedback_str,
            success=success,
            error_value=current_error
        )
        results_aggregator.add_result(iter_result)
        

        # Check for early stopping (enhanced logic from reference_code.py)
        should_stop, stop_reason = results_aggregator.should_early_stop(
            target_error=args.target_error,
            min_iterations=2,
            no_improvement_patience=args.no_improvement_patience
        )

        if should_stop:
            print(f"\n[Early Stop] {stop_reason}")
            break

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("HA-SCIENTIST LOOP COMPLETE")
    print("="*80)

    if results_aggregator.best_result:
        print(f"Best Error Achieved: {results_aggregator.best_error:.6f}")
        print(f"Best Iteration: {results_aggregator.best_result.iteration}")

        # Print summary of all iterations
        print("\n--- Iteration Summary ---")
        for res in results_aggregator.results:
            status = "✓" if res.success else "✗"
            error_str = f"{res.error_value:.6f}" if res.error_value < float('inf') else "N/A"
            best_marker = " (BEST)" if res == results_aggregator.best_result else ""
            print(f"  Iter {res.iteration}: [{status}] Error={error_str}{best_marker}")
    else:
        print("No successful results achieved.")
        print(f"Total iterations attempted: {len(results_aggregator.results)}")

    # Final Evaluation of the best result (saved to run folder)
    if results_aggregator.best_result and results_aggregator.best_result.ha_specification:
        print("\n--- Final Evaluation of Best Result ---")
        # Save the best HA specification to the run directory
        best_spec_dir = os.path.join("evaluation_results", relative_data_path, "runs", run_id)
        best_spec_path = os.path.join(best_spec_dir, "best_ha_specification.json")
        os.makedirs(best_spec_dir, exist_ok=True)
        with open(best_spec_path, "w", encoding="utf-8") as f:
            json.dump(results_aggregator.best_result.ha_specification, f, indent=2)
        print(f"Best HA specification saved to: {best_spec_path}")
        
        # Copy the best iteration folder as best_iter_*
        best_iter_num = results_aggregator.best_result.iteration
        source_iter_dir = os.path.join(best_spec_dir, f"iter_{best_iter_num}")
        dest_iter_dir = os.path.join(best_spec_dir, f"best_iter_{best_iter_num}")
        if os.path.exists(source_iter_dir):
            if os.path.exists(dest_iter_dir):
                shutil.rmtree(dest_iter_dir)  # Remove existing if present
            shutil.copytree(source_iter_dir, dest_iter_dir)
            print(f"Best iteration folder copied to: {dest_iter_dir}")
        else:
            print(f"Warning: Source iteration folder not found: {source_iter_dir}")
        
        print(f"All artifacts available at: evaluation_results/{relative_data_path}/runs/{run_id}/")



if __name__ == "__main__":
    main()

    
