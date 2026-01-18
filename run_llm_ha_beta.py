import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
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
from utils.Dainarx_code.HA_evaluation import HAEvaluator
from utils.ha_spec_validator import preprocess_ha_for_evaluation
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
                     model_id: str = "gemini/gemini-3-flash-preview",
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

## AVAILABLE DATA FILES
You have access to the following NPZ data files:
{npz_files_description}

**Quick Access via State Variables:**
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `DATA_FILE_PATH`: Path to the first/primary data file (for convenience)

**Example Usage:**
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
def create_agent(model_id: str = "gemini/gemini-3-flash-preview",
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
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
A Hybrid Automaton models a system with:
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each regime)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)


## Quality Criteria
Your HA specification will be evaluated on:
- **Trajectory Matching**: Simulated output should closely follow ground truth data
- **Mode Detection Accuracy**: Correct identification of switching instants (TC (Change-Point Error) < 0.01s is good, <= 0.001s is excellent)
- **State Error Minimization**: Low Mean Difference (Mean Difference) and Maximum Difference (Max Difference) between predicted and actual states (Max Difference < 0.005 is good, < 0.0001 is excellent)"""

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
\n**Sub-Agents Resources:** You have access to managed Code Agent(s): `{managed_agents_list}`. The `{managed_agents_list}` have access to the npz data files and can be used to analyze the data."""
        task += MANAGE_AGENT_PROMPT

    # Add self code agent prompt
    if manager_type == "CodeAgent":
        SELF_IS_CODE_AGENT_PROMPT = "\n## Code Execution Capability\nYou can use Python Code to execute programs, which may help with your task-solving process."
        task += SELF_IS_CODE_AGENT_PROMPT

    # Generate dynamic HA template with correct var and input fields pre-filled
    dynamic_ha_template = generate_dynamic_ha_template(num_variables, num_inputs)
    
    # Generate variable and input names for display
    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else "(none)"

    # Add HA specification format documentation and initial spec
    # Use JSON Schema-based documentation for more precise output specification
    # We use the explicit schema and examples to ensure consistency with the improved task prompt
    ha_spec_docs = """## Hybrid Automaton Specification Format (JSON Schema)

### JSON Schema Definition
Your output MUST conform to this JSON Schema:

```json
{
  "title": "Hybrid Automaton Specification",
  "description": "Schema for defining hybrid automaton systems with modes, transitions, and configuration",
  "type": "object",
  "required": [
    "automaton",
    "config"
  ],
  "additionalProperties": true,
  "properties": {
    "automaton": {
      "type": "object",
      "description": "The hybrid automaton definition with variables, modes, and transitions",
      "required": [
        "var",
        "mode",
        "edge"
      ],
      "properties": {
        "var": {
          "type": "string",
          "description": "Comma-separated list of state variable names (e.g., 'x1' or 'x1, x2')",
          "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\\\\s*,\\\\s*[a-zA-Z_][a-zA-Z0-9_]*)*$",
          "minLength": 1,
          "examples": [
            "x1",
            "x1, x2",
            "x, y, z"
          ]
        },
        "input": {
          "type": "string",
          "description": "Comma-separated list of input variable names (e.g., 'u1' or 'u1, u2'). Empty string if no inputs.",
          "pattern": "^([a-zA-Z_][a-zA-Z0-9_]*(\\\\s*,\\\\s*[a-zA-Z_][a-zA-Z0-9_]*)*)?$",
          "default": "",
          "examples": [
            "",
            "u1",
            "u1, u2"
          ]
        },
        "mode": {
          "type": "array",
          "description": "List of discrete modes (operating regimes) in the automaton",
          "minItems": 1,
          "items": {
            "$ref": "#/definitions/mode"
          }
        },
        "edge": {
          "type": "array",
          "description": "List of discrete transitions between modes (can be empty [])",
          "items": {
            "$ref": "#/definitions/edge"
          }
        }
      },
      "additionalProperties": false
    },
    "config": {
      "type": "object",
      "description": "Simulation and learning configuration parameters",
      "required": [
        "dt",
        "total_time",
        "order"
      ],
      "properties": {
        "dt": {
          "type": "number",
          "description": "Integration time step in seconds",
          "exclusiveMinimum": 0,
          "examples": [
            0.001,
            0.01
          ]
        },
        "total_time": {
          "type": "number",
          "description": "Total simulation duration in seconds",
          "exclusiveMinimum": 0,
          "examples": [
            10.0,
            20.0
          ]
        },
        "order": {
          "type": "integer",
          "description": "Order of the ODE system (1=first-order, 2=second-order, etc.)",
          "minimum": 1,
          "default": 1
        },
        "need_reset": {
          "type": "boolean",
          "description": "Whether variable resets occur on mode transitions, if there are more than 2 modes, the reset is required",
          "default": false
        },
        "non_linear_items": {
          "type": "string",
          "description": "Nonlinear/cross terms in dynamics (e.g., 'x1[?]**3', 'x1[?]*x2[?]') in ODEs, which is eq in mode. The nonlinear/cross terms MUST be consistent with the 'eq' in mode.",
          "default": ""
        },
        "self_loop": {
          "type": "boolean",
          "description": "Whether self-loop transitions are allowed",
          "default": false
        },
        "need_bias": {
          "type": "boolean",
          "description": "Whether to include constant term in ODEs"
        }
      },
      "additionalProperties": true
    }
  },
  "definitions": {
    "mode": {
      "type": "object",
      "description": "A discrete mode (operating regime) with its continuous dynamics",
      "required": [
        "id",
        "eq"
      ],
      "properties": {
        "id": {
          "type": "integer",
          "description": "Unique mode identifier (must be >= 1)",
          "minimum": 1
        },
        "eq": {
          "type": "string",
          "description": "ODE equation(s) defining the continuous dynamics. Format: 'var[order] = expression'. Multiple equations separated by commas.",
          "minLength": 1,
          "examples": [
            "x1[1] = -2 * x1[0] + u1",
            "x1[2] = x1[1] - x1[0] ** 2 + u1"
          ]
        }
      },
      "additionalProperties": false
    },
    "edge": {
      "type": "object",
      "description": "A discrete transition (edge) between modes",
      "required": [
        "direction",
        "condition"
      ],
      "properties": {
        "direction": {
          "type": "string",
          "description": "Transition direction in format 'source -> target' (e.g., '1 -> 2')",
          "pattern": "^\\\\d+\\\\s*->\\\\s*\\\\d+$"
        },
        "condition": {
          "type": "string",
          "description": "Guard condition that triggers the transition. Use bare variable names (x1, not x1[0]).",
          "minLength": 1,
          "examples": [
            "x1 > 0.5",
            "x1 <= 0 and x2 > 1",
            "abs(x1) >= 1.2"
          ]
        },
        "reset": {
          "type": "object",
          "description": "Optional state reset map. Keys are variable names, values are arrays of reset expressions.",
          "additionalProperties": {
            "type": "array",
            "description": "Reset values for each derivative order: [x[0], x[1], ...]. Use '' to preserve current value.",
            "items": {
              "oneOf": [
                {
                  "type": "string"
                },
                {
                  "type": "number"
                }
              ]
            }
          },
          "examples": [
            {
              "x1": [
                0
              ],
              "x2": [
                "-0.9 * x2[0]"
              ]
            },
            {
              "x": [
                "",
                "x[1] * 0.95"
              ]
            }
          ]
        }
      },
      "additionalProperties": false
    }
  }
}
```


### Examples

**Single-mode 2nd-order:**
```json
{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = x1[1] + x1[0] + x1[0] ** 2 + u1"
            }
        ],
        "edge": []
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 2,
        "need_reset": false,
        "non_linear_items": "x1[?] ** 2"
    }
}
```

**Two-mode switching:**
```json
{
    "automaton": {
        "var": "x1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x1[0] + 5"
            },
            {
                "id": 2,
                "eq": "x1[1] = -x1[0] - 3"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "x1 >= 4"
            },
            {
                "direction": "2 -> 1",
                "condition": "x1 <= 9"
            }
        ]
    },
    "config": {
        "dt": 0.01,
        "total_time": 20.0,
        "order": 1,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

**With reset map:**
```json
{
    "automaton": {
        "var": "x1, x2",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = x1[0] + u1"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {
                    "x1": [0],
                    "x2": ["-0.1 * x1[0]"]
                }
            }
        ]
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 1,
        "need_reset": true,
        "non_linear_items": ""
    }
}
```"""

    task += f"""
{ha_spec_docs}

## ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification (v0)
The `var` and `input` fields are pre-filled.

```json
{dynamic_ha_template}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "{var_names}"` and `input: "{input_names if num_inputs > 0 else ''}"` exactly as shown!**
- Make sure the HA specification is valid and complete.
- Refine the HA specification to improve trajectory matching and reduce TC (Change-Point Error), Mean Difference, and Maximum Difference.

## Available Data
The following data sources are provided:
- **Trace visualizations**: {image_placeholders}, use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.
- **Raw data files**: {npz_placeholders}. If you want to use the npz data to analyze the system, you MUSTuse the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.
"""

    # Add feedback from previous iteration if available
    if feedback:
        task += f"""
## ⚠️ FEEDBACK FROM PREVIOUS ITERATION
The following feedback was generated from evaluating your previous attempt. Use it to guide your next refinement:
    
    {feedback}
    
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
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # default_data_path = os.path.join(script_dir, "utils", "Dainarx_code", "data_duffing")

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
        default="gemini/gemini-3-flash-preview",
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
        default="gemini-3-flash-preview",
        help="Model ID to use for the image analysis tool (Gemini API format).",
    )
    ap.add_argument(
        "--review-tool-model",
        type=str,
        default="gemini-3-flash-preview",
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
        default=['data_analysis_expert'],
        help="List of managed agents to use in the agent.",
    )

    # LLM model ids for the managed agents
    ap.add_argument(
        "--managed-agents-list-model",
        type=str,
        default="gemini/gemini-3-flash-preview",
        help="Model ID to use for managed agents.",
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

    args = ap.parse_args()

    if not args.input_data_path:
        raise ValueError("You must provide a data path with --input-data-path.")
    if not os.path.exists(args.input_data_path):
        raise FileNotFoundError(f"Data path {args.input_data_path} does not exist.")

    return args


def main():
    args = parse_args()
    print(f"Running HA Learning Agent with model: {args.manager_model}, tools: {args.tools_list},managed agents: {args.managed_agents_list}, data path: {args.input_data_path}")

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
        feedback_min_gap=args.feedback_min_gap,
        target_error=args.target_error,
        no_improvement_patience=args.no_improvement_patience,
        input_data_path=args.input_data_path,
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
        managed_agents_list=args.managed_agents_list if hasattr(args, 'managed_agents_list') else None,
        managed_agents_list_model_id=args.managed_agents_list_model if hasattr(args, 'managed_agents_list_model') else None,
        manager_type=args.manager_type,
        tools_to_remove=args.tools_to_remove,
    )

    # ========================================================================
    # HA-Scientist Iterative Loop (Enhanced with ResultsAggregator)
    # Inspired by SR-Scientist\inference\infer\inference.py's multi-turn adaptive loop pattern
    # ========================================================================

    # Initialize results aggregator for intelligent feedback selection
    results_aggregator = ResultsAggregator(
        top_k=args.feedback_top_k,
        min_gap=args.feedback_min_gap
    )

    print(f"\nSTARTING HA-SCIENTIST LOOP (Max iterations: {args.max_iterations})")
    print(f"  - Top-K feedback selection: {results_aggregator.top_k}")
    print(f"  - Diversity gap threshold: {results_aggregator.min_gap}")
    print(f"  - Target error for early stop: {args.target_error}")
    print(f"  - No-improvement patience: {args.no_improvement_patience}")

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

            # Also include latest iteration's feedback for recency
            latest_feedback = results_aggregator.get_latest_feedback()
            if latest_feedback and latest_feedback not in current_feedback:
                current_feedback += f"\n## Most Recent Attempt (Iteration {iteration - 1}):\n{latest_feedback}"

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
            print(f"Agent execution failed: {e}")
            # Create failed iteration result
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
            output_dir=os.path.join("evaluation_results", f"iter_{iteration}"),
            hyperparameters=hyperparameters,
            iteration=iteration
        )

        # Extract error value from metrics
        current_error = float('inf')
        if success and isinstance(metrics, dict):
            # Priority order for error metrics
            if 'max_diff' in metrics:
                current_error = metrics['max_diff']
            elif 'mean_diff' in metrics:
                current_error = metrics['mean_diff']
            elif 'tc' in metrics:
                current_error = metrics['tc']
            elif 'rmse' in metrics:
                current_error = metrics['rmse']
            

        print(f"Iteration {iteration} Result Error: {current_error}")

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

    # Final Evaluation of the best result (saved to main evaluation folder)
    if results_aggregator.best_result and results_aggregator.best_result.ha_specification:
        print("\n--- Final Evaluation of Best Result ---")
        # Save the best HA specification to a JSON file
        best_spec_path = os.path.join("evaluation_results", "best_ha_specification.json")
        os.makedirs("evaluation_results", exist_ok=True)
        with open(best_spec_path, "w", encoding="utf-8") as f:
            json.dump(results_aggregator.best_result.ha_specification, f, indent=2)
        print(f"Best HA specification saved to: {best_spec_path}")


if __name__ == "__main__":
    main()

    # Example usage:
    # python run_llm_ha_beta.py --input-data-path data_all/non_linear/duffing --manager-type CodeAgent --tools-list hybrid_automaton_image_analysis summarize_hybrid_automaton_iterations validate_hybrid_automaton_specification
