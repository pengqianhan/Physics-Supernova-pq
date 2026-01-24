import os
import sys
import json
import secrets
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
        "dt": 0.001,
        "total_time": 10.0,
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

    return managerAgent


def get_managed_agents_list(managed_agents_list: List[str] = None,
                            managed_agents_list_model_id: str = None,
                            input_data_path: str = None,
                            markdown_content: MarkdownMessage = None) -> List[MultiStepAgent]:
    if managed_agents_list is None:
        return []

    # Load trace data to get NPZ paths if not provided
    if markdown_content is None:
        markdown_content = load_trace_data_from_filepath(input_data_path)

    # Get NPZ file paths from markdown_content
    # npz_paths is a dict like {"<image_0>": "/path/to/sample_0.npz", ...}
    npz_paths_list = list(markdown_content.npz_paths.values()) if markdown_content.npz_paths else []
    npz_placeholders = list(markdown_content.npz_paths.keys()) 

    # Fallback to default path if no npz files found
    if not npz_paths_list:
        npz_paths_list = [os.path.join(input_data_path, "sample_0.npz")]

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

        # Build file paths description for prompt
        npz_files_description = "\n".join([f" {placeholders} - `{path}`" for placeholders, path in zip(npz_placeholders, npz_paths_list)])

        # managed agent description with all available files
        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can assist with code-related tasks.

## Available npz files
You have access to the following NPZ data files:
{npz_files_description}

**Quick Access via State Variables**
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `DATA_FILE_PATH`: Path to the first/primary data file (for convenience)

**Example Usage**
```python
import numpy as np
# Load a specific file
data = np.load(DATA_FILE_PATHS[0])
# Or use the primary file
data = np.load(DATA_FILE_PATH)
```

"""
        # use_e2b = bool(os.environ.get("E2B_API_KEY"))
        use_e2b = False
        # save the managed_agent_description to a file
        with open(f"managed_agent_description_{agent_name}.md", "w") as f:
            f.write(managed_agent_description)
        managed_agent = CodeAgent(
            tools=[],
            executor_type="e2b" if use_e2b else "local",
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
            # 
            "pysindy","gradient_free_optimizers","gradient_free_optimizers.BayesianOptimizer"
        ],
            description=managed_agent_description,
            max_steps=80,
            verbosity_level=2,
        )
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
            # 注入所有文件路径到 agent 状态
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = sandbox_file_paths
            managed_agent.python_executor.state["DATA_FILE_PATH"] = sandbox_file_paths[0] if sandbox_file_paths else ""
        else:
            # 本地执行器：将所有数据文件路径注入到 agent 的状态中
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = npz_paths_list
            managed_agent.python_executor.state["DATA_FILE_PATH"] = npz_paths_list[0] if npz_paths_list else ""
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
        managed_agents_list=get_managed_agents_list(
            managed_agents_list,
            managed_agents_list_model_id,
            input_data_path,
            markdown_content=markdown_content  # Pass markdown_content to reuse loaded npz paths
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
                           use_json_schema: bool = True) -> tuple[str, list]:
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

    Returns:
        Tuple of (task prompt string, list of compressed images)
    '''
    print(f"Generating task for iteration {iteration}, tools_list: {tools_list}")

    # Load trace data with high res images
    markdown_content = load_trace_data_from_filepath(input_data_path)
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
- `Max Difference < 0.01`: good fit
- `Mean Difference < 0.005`: accurate overall
- `TC (Change-Point Error) < 0.01s`: mode switch timing correct
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

    # # Add self code agent prompt
    # if manager_type == "CodeAgent":
    #     SELF_IS_CODE_AGENT_PROMPT = "\n## Code Execution Capability\nYou can use Python Code to execute programs, which may help with your task-solving process."
    #     task += SELF_IS_CODE_AGENT_PROMPT

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
- ODEnotation:
   - `x[0]` = variable value
   - `x[1]` = first derivative (dx/dt)
   - `x[2]` = second derivative (d²x/dt²)
   - Left side = highest derivative (e.g., `x1[2] = ...` for order=2)
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)

## Initial Hybrid Automaton Specification (v0)

```json
{dynamic_ha_template}
```

## Your Task
Generate an improved HA specification that better matches the observed trajectory data.
- **Keep `var: "{var_names}"` and `input: "{input_names if num_inputs > 0 else ''}"` exactly as shown!**
- Make sure the HA specification is valid and complete according to the JSON Schema.
- Refine the HA specification to improve trajectory matching and reduce `Max Difference`, `Mean Difference`, and `TC (Change-Point Error)`.

## Available Data
- **Trace visualizations**: {image_placeholders}, use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.
- **Raw data files**: {npz_placeholders}. If you want to use the npz data to analyze the system, you MUST use the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.

"""

    # Add feedback from previous iteration if available
    if feedback:
        task += f""""\n## Previously Explored HA Specifications with Feedback",
            "Use these as inspiration to guide your next refinement.",
            "Use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section."
            {feedback}"""


    return task, compressed_trace_images


def gen_summary(
    metrics_dict: Dict,
    ha_specification: Dict,
    plot_path: str,
    model_id: str = "gemini/gemini-3-flash-preview"
) -> str:
    """
    Generate a comprehensive summary of evaluation results using LLM with vision.

    Analyzes the metrics, HA specification, and evaluation plot to provide
    actionable feedback for improving the hybrid automaton model.

    Args:
        metrics_dict: Dictionary containing evaluation metrics (mean_diff, max_diff, tc, etc.)
        ha_specification: The HA specification dictionary being evaluated
        plot_path: Path to the overlay plot image (ground truth vs simulated)
        model_id: LiteLLM model ID (default: gemini/gemini-3-flash-preview)

    Returns:
        LLM-generated summary with analysis and improvement suggestions
    """


    # Fallback if no metrics
    if not metrics_dict:
        return "No metrics available for analysis."

    # Read and encode the plot image
    image_content = None
    if plot_path and os.path.isfile(plot_path):
        try:
            with open(plot_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            # Determine mime type from extension
            ext = os.path.splitext(plot_path)[1].lower()
            mime_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            }
        except Exception as e:
            print(f"[gen_summary] Warning: Could not read plot image: {e}")

    # Format metrics for prompt
    metrics_text = "\n".join([
        f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}"
        for k, v in metrics_dict.items() if v is not None
    ])

    # Format HA spec summary (truncate if too long)
    ha_spec_str = json.dumps(ha_specification, indent=2)
    if len(ha_spec_str) > 2000:
        ha_spec_str = ha_spec_str[:2000] + "\n... (truncated)"

    # Build the analysis prompt
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
Analyze the evaluation results and the trajectory comparison plot (ground truth vs simulated).
Identify the main sources of error and suggest specific improvements to the ODE equations or guard conditions."""

    # Build message content
    content = [{"type": "text", "text": user_prompt}]
    if image_content:
        content.insert(0, image_content)

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
    plot_placeholder: str,
    plot_summary: str
) -> Dict:
    """
    Build the artifact manifest structure for feedback.

    Args:
        run_id: Unique identifier for this experiment run
        iteration: Current iteration number
        metrics: Evaluation metrics dictionary
        plot_placeholder: Placeholder reference for the overlay plot (e.g., <iter_image_1>)
        plot_summary: Text summary for fallback

    Returns:
        Dictionary containing the artifact manifest
    """
    return {
        "Evaluation Metrics": {
            "TC (Change-Point Error)": metrics.get('tc', None),
            "Max Difference": metrics.get('max_diff', None),
            "Mean Difference": metrics.get('mean_diff', None),
        },
        "Evaluation Plot": {
            "Placeholder": plot_placeholder,
            "Summary": plot_summary,
        }
    }


def evaluate_ha_specification_with_feedback(
    agent_result,
    input_data_path: str,
    output_dir: str = None,
    hyperparameters: HAHyperparameters = None,
    iteration: int = 1,
    use_structured_output: bool = True,
    structured_output_model: str = "gemini-3-flash-preview",
    run_id: str = None,
    summary_model: str = "gemini-3-flash-preview",
    image_tool: HybridAutomatonImageTool = None
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

    # Find test data file
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

    npz_file_path = os.path.join(test_data_base_path, test_data_files[0])
    
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

        # Run evaluation - use consistent file names for artifact referencing
        # Save overlay.png (consistent name) for artifact manifest
        overlay_path = os.path.abspath(os.path.join(output_dir, 'overlay.png'))
        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path=overlay_path,
            print_metrics=True
        )
        metrics_dict = evaluator.metrics

        # Save metrics and hyperparameters to metrics.txt (consistent name)
        metrics_file = os.path.join(output_dir, 'metrics.txt')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Run ID: {run_id if run_id else 'N/A'}\n\n")
            f.write(f"Metrics:\n{metrics_text}\n\n")
            f.write(f"HA Specification:\n{json.dumps(ha_specification, indent=2)}\n\n")

            # Write hyperparameters if provided
            if hyperparameters is not None:
                f.write(hyperparameters.to_string())
                f.write("\n\n")

        # Generate plot summary using LLM analysis of metrics, HA spec, and plot image
        plot_summary = gen_summary(metrics_dict, ha_specification, overlay_path, model_id=summary_model)

        # Register the overlay image with the image tool for placeholder-based access
        # Use iteration number as the image index (e.g., iteration 1 -> <iter_image_1>)
        plot_placeholder = f"<iter_image_{iteration}>"
        if image_tool is not None:
            success, error_msg = image_tool.register_iteration_image(iteration, overlay_path)
            if not success:
                print(f"[Warning] Failed to register overlay image: {error_msg}")
        else:
            print("[Warning] No image_tool provided - evaluation plots won't be available via placeholder")

        # Build artifact manifest with placeholder instead of path
        artifact_manifest = build_artifact_manifest(
            run_id=run_id if run_id else "unknown",
            iteration=iteration,
            metrics=metrics_dict,
            plot_placeholder=plot_placeholder,
            plot_summary=plot_summary
        )

        # Save artifact manifest to disk (for debugging/external tools)
        manifest_file = os.path.join(output_dir, 'artifacts.json')
        with open(manifest_file, 'w') as f:
            json.dump(artifact_manifest, f, indent=2)

        # Construct feedback string with HA spec, metrics, and artifact manifest
        feedback = f"The {iteration}th attempt result:\n"
        feedback += f"  1. HA JSON Specification:\n```json\n{json.dumps(ha_specification, indent=2)}\n```\n"
        feedback += f"  2. Evaluation Feedback:\n{json.dumps(artifact_manifest, indent=2)}\n"

        return True, metrics_dict, feedback, ha_specification

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

    args = ap.parse_args()

    if not args.input_data_path:
        raise ValueError("You must provide a data path with --input-data-path.")
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


def main():
    args = parse_args()
    print(f"Running HA Learning Agent with model: {args.manager_model}, tools: {args.tools_list},managed agents: {args.managed_agents_list}, data path: {args.input_data_path}")

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

            # Show dynamic target if available
            dynamic_target = results_aggregator.get_dynamic_error_threshold()
            print(f"  [Aggregator] Dynamic error target: {dynamic_target:.6f}")

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
            use_json_schema=args.use_json_schema
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
            image_tool=image_tool
        )

        # Extract error value from metrics
        current_error = float('inf')
        if success and isinstance(metrics, dict) and 'mean_diff' in metrics:
            current_error = metrics['mean_diff']
            print(f"Iteration {iteration} mean_diff: {current_error:.6f}, max_diff: {metrics.get('max_diff', float('inf')):.6f}, tc: {metrics.get('tc', float('inf')):.6f}")
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
        print(f"All artifacts available at: evaluation_results/{relative_data_path}/runs/{run_id}/")



if __name__ == "__main__":
    main()

    
