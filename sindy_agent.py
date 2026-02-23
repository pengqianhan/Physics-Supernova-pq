
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
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress

import argparse
from PIL import Image

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
from utils.local_image_qa_tool import LocalImageQATool
from utils.reviewTools_ha import ReviewRequestTool_ha
from utils.summemoryTools_ha import SummarizeMemoryTool
from utils.validateTools_ha import ValidateHASpecTool
from utils.Dainarx_code.HA_evaluation import HAEvaluator
from utils.ha_spec_validator import preprocess_ha_for_evaluation
from utils.ha_structured_output import preprocess_ha_for_evaluation_v2, convert_agent_result_to_ha
from utils.sindy_code_template import sindy_code_example
import numpy as np


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
    "pysindy.optimizers", "pysindy.feature_library", "pysindy.differentiation",
]


def get_litellm_kwargs(model_id: str) -> dict:
    """
    Return the appropriate api_key and api_base for a given model_id.

    Supports:
    - moonshot/*, openai/kimi*: Moonshot API (MOONSHOT_API_KEY)
    - gemini/*: Google Gemini API (GEMINI_API_KEY)
    - Other providers: rely on litellm's env-var auto-detection
    """
    if not model_id:
        return {}

    if (
        model_id.startswith("moonshot/")
        or model_id.startswith("openai/kimi")
        or model_id.startswith("openai/moonshot")
        or model_id.startswith("kimi")
    ):
        return {
            "api_key": os.environ.get("MOONSHOT_API_KEY"),
            "api_base": os.environ.get("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
        }
    elif model_id.startswith("gemini/"):
        return {
            "api_key": os.environ.get("GEMINI_API_KEY"),
        }
    else:
        # For other providers (openrouter, anthropic, etc.), let litellm handle it
        return {}
input_data_path = "data_all/non_linear/duffing"
train_num = 3
markdown_content = load_trace_data_from_filepath(input_data_path, train_num=train_num)
managed_agents_list = ["sindy_agent"]
managed_agents_list_model_id = "openai/kimi-k2.5"
dataset_subpath = input_data_path.replace("data_all/", "", 1) if input_data_path.startswith("data_all/") else input_data_path
image_plot_path_list = [os.path.join("analysis_plots_train", dataset_subpath, f"sample_{i}_analysis.png") for i in range(train_num)]

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

    dataset_subpath = input_data_path.replace("data_all/", "", 1) if input_data_path.startswith("data_all/") else input_data_path
    image_plot_path_list = [os.path.join("analysis_plots_train", dataset_subpath, f"sample_{i}_analysis.png") for i in range(train_num)]

    # Fallback to default path if no npz files found
    if not npz_paths_list:
        npz_paths_list = [os.path.join(input_data_path, "sample_0.npz")]

    managed_agents = []
    llm_kwargs = get_litellm_kwargs(managed_agents_list_model_id)
    model = LiteLLMModel(
            model_id=managed_agents_list_model_id,
            **llm_kwargs,
            # max_completion_tokens=24576,
            num_retries=3,
            timeout=1200,
        )
    managed_agent_kwargs = dict(
        model=model,
        tools=[],
        max_steps=80,
        verbosity_level=2,
        add_base_tools=True,
        additional_authorized_imports=AUTHORIZED_IMPORTS_LIST,
    )
    managed_image_tool_model_id = llm_kwargs["model_id"] 
    if managed_agents_list_model_id and managed_agents_list_model_id.startswith("gemini/"):
        managed_image_tool_model_id = managed_agents_list_model_id
    npz_files_description = "\n".join([f" {placeholder} - `{path}` - `DATA_FILE_PATHS[{i}]`" for i, (placeholder, path) in enumerate(zip(npz_placeholders, npz_paths_list))])
    image_plot_description = "\n".join([f" `{path}` - `IMAGE_PLOT_PATHS[{i}]`" for i, path in enumerate(image_plot_path_list)])
    for agent_name in managed_agents_list:

        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can analyze the change points and dynamics of the data from the image plot. According to the change points, segment the data into different segments, these segments are saved in the a list called "SEG_DATA_LIST". For each segment, fit the data using SINDy, the SINDy model is saved in the a list called "SEG_MODEL_LIST". If the SINDy model is similar between different segments, merge the SINDy model into one. Return the SINDy model for the whole data."""
        common_instruction = """
### Quick Access via State Variables
#### Available Files
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `IMAGE_PLOT_PATHS`: List of all available image plot file paths
Each image plot contains the following subplots (top to bottom), all sharing the same x-axis (time in seconds):

For each state variable xi:
- **Position** (row 1): raw state value `state[i]` from the NPZ file
- **Velocity** (row 2): first derivative `dx_i/dt`, computed via `np.gradient(state[i], dt)`
- **Acceleration** (row 3): second derivative `d²x_i/dt²`, computed via `np.gradient(velocity, dt)`

After all state variables, if inputs exist:
- **Input** (final rows): each input signal `u1, u2, ...` from the NPZ `input` array

Use these plots to identify change points (abrupt changes in velocity or acceleration indicate mode transitions).
#### Available Tools
- `local_image_qa(image_path, question)`: Analyze the change points and dynamics of the data


### Guidelines
1. Analyze the change points and dynamics of the data from the image plot.
2. According to the change points, segment the data into different segments, these segments are saved in the a list called "SEG_DATA_LIST".
3. For each segment, fit the data using SINDy, the SINDy model is saved in the a list called "SEG_MODEL_LIST".
```python
SEG_MODEL_LIST = []
for seg_data in SEG_DATA_LIST:
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    # feature_names is a parameter of fit(), NOT __init__()
    # u= is optional control input; pass it if the data has external forcing
    model.fit(seg_data, t=dt, feature_names=['x1', 'x2'], u=control_input)
    model.print()
    SEG_MODEL_LIST.append(model)
```
4. If the SINDy model is similar between different segments, merge them.
   Compare the coefficient matrices (`model.coefficients()`) between segments.
   If two segments have similar coefficients (e.g., small Frobenius norm difference),
   treat them as the same mode and re-fit a single SINDy model on their combined data.

5. Return the SINDy model for each distinct mode.

### Example Usage

```python
import os
import numpy as np

#### Load the data (DATA_FILE_PATHS is a list; index to get a single path)
data = np.load(DATA_FILE_PATHS[0])

#### Load the image plot file (IMAGE_PLOT_PATHS is a list of paths)
image_plot_path = IMAGE_PLOT_PATHS[0]

#### Ask a vision model about the data
answer = local_image_qa(
    image_path=image_plot_path,
    question="What are the change points of the data?"
)
```
#### Here is the example of the SINDy model usage:
```python
__SINDY_CODE_EXAMPLE__
```

"""
        common_instruction = common_instruction.replace("__SINDY_CODE_EXAMPLE__", sindy_code_example)
        # save the common_instruction to file
        with open(f"common_instruction_{agent_name}.md", "w") as f:
            f.write(common_instruction)
        managed_tools = []
        if agent_name == "sindy_agent":
            managed_tools.append(LocalImageQATool(model_id=managed_image_tool_model_id))
        # Build instruction (shared by both E2B and local executors)
        managed_agent_instruction = f"""## Available NPZ Files

You have access to the following NPZ data files:

{npz_files_description}
and the following image plot files:
{image_plot_description}

""" + common_instruction

        # use_e2b = bool(os.environ.get("E2B_API_KEY"))
        use_e2b = False
        if use_e2b:
            print("Using E2B cloud sandbox executor, uploading data files...")
            # Upload all files to E2B sandbox
            sandbox_file_paths = []
            for i, npz_path in enumerate(npz_paths_list):
                with open(npz_path, "rb") as f:
                    file_content = f.read()
                sandbox_file_path = f"/tmp/sample_{i}.npz"
                managed_agent.python_executor.sandbox.files.write(sandbox_file_path, file_content)
                sandbox_file_paths.append(sandbox_file_path)
                print(f"  File uploaded to E2B sandbox: {sandbox_file_path}")
            managed_agent_kwargs["executor_type"] = "e2b"
            managed_agent_kwargs["name"] = agent_name
            managed_agent_kwargs["description"] = managed_agent_description
            managed_agent_kwargs["instructions"] = managed_agent_instruction
            managed_agent_kwargs["tools"] = managed_tools
            managed_agent = CodeAgent(**managed_agent_kwargs)
            # Inject all file paths into agent state
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = sandbox_file_paths
            managed_agent.python_executor.state["DATA_FILE_PATH"] = sandbox_file_paths[0] if sandbox_file_paths else ""
            managed_agent.python_executor.state["IMAGE_PLOT_PATHS"] = image_plot_path_list
        else:
            # Local executor: inject all data file paths into agent state
            managed_agent_kwargs["name"] = agent_name
            managed_agent_kwargs["executor_type"] = "local"
            managed_agent_kwargs["description"] = managed_agent_description
            managed_agent_kwargs["instructions"] = managed_agent_instruction
            managed_agent_kwargs["tools"] = managed_tools
            managed_agent = CodeAgent(**managed_agent_kwargs)
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = npz_paths_list
            managed_agent.python_executor.state["IMAGE_PLOT_PATHS"] = image_plot_path_list
        managed_agents.append(managed_agent)

    return managed_agents

managed_agents = get_managed_agents_list(managed_agents_list, managed_agents_list_model_id, input_data_path, markdown_content, train_num)
print(managed_agents)

# run the managed agents
task = "Analyze the change points and dynamics of the data from the image plot. According to the change points, segment the data into different segments. For each segment, fit the data using SINDy. Return the SINDy model for each segment. If the SINDy model is similar between different segments, merge the SINDy model into one. Return the SINDy model for the whole data."

agent_images = []
for image_plot_path in image_plot_path_list:
    with Image.open(image_plot_path) as img:
        # smolagents expects PIL-like objects for `images`, not raw bytes.
        agent_images.append(img.copy())

for managed_agent in managed_agents:
    managed_agent.run(task=task, images=agent_images)