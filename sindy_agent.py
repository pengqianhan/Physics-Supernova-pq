
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
image_plot_path_list =  ['analysis_plots_train/non_linear/duffing/sample_0_analysis.png','analysis_plots_train/non_linear/duffing/sample_1_analysis.png','analysis_plots_train/non_linear/duffing/sample_2_analysis.png']
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

    image_plot_path_list =  ['analysis_plots_train/non_linear/duffing/sample_0_analysis.png','analysis_plots_train/non_linear/duffing/sample_1_analysis.png','analysis_plots_train/non_linear/duffing/sample_2_analysis.png']

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
    managed_image_tool_model_id = "gemini/gemini-flash-lite-latest"
    if managed_agents_list_model_id and managed_agents_list_model_id.startswith("gemini/"):
        managed_image_tool_model_id = managed_agents_list_model_id
    npz_files_description = "\n".join([f" {placeholder} - `{path}` - `DATA_FILE_PATHS[{i}]`" for i, (placeholder, path) in enumerate(zip(npz_placeholders, npz_paths_list))])
    image_plot_description = "\n".join([f" {path}` - `IMAGE_PLOT_PATHS[{i}]`" for i, path in enumerate(image_plot_path_list)])
    for agent_name in managed_agents_list:

        managed_agent_description = f"""I am a managed agent with name {agent_name}. I can assist with code-related tasks."""
        common_instruction = """
### Quick Access via State Variables
#### Available Files
- `DATA_FILE_PATHS`: List of all available NPZ file paths
- `IMAGE_PLOT_PATHS`: List of all available image plot file paths
The plot image is drawn by these code 
```python 
def analyze_dataset(dataset, sample_idx, output_root):
    data_dir = os.path.join(DATA_ROOT, dataset)
    npz_path = os.path.join(data_dir, f"sample_{sample_idx}.npz")
    if not os.path.exists(npz_path):
        print(f"  [SKIP] {npz_path} not found")
        return

    data = np.load(npz_path)
    state = data["state"]       # (num_vars, num_steps)
    inp = data["input"]         # (num_inputs, num_steps) or (num_steps,)

    num_vars = state.shape[0]
    num_steps = state.shape[1]

    # Ensure input is 2D
    if inp.ndim == 1:
        inp = inp.reshape(1, -1)
    num_inputs = inp.shape[0]
    has_input = num_inputs > 0 and inp.size > 0

    # Load config for dt and var names
    cfg = load_config(dataset)
    if cfg:
        dt = cfg["dt"]
        var_names = cfg["var_names"]
    else:
        dt = 0.01
        var_names = [f"x{i+1}" for i in range(num_vars)]

    while len(var_names) < num_vars:
        var_names.append(f"x{len(var_names)+1}")

    t = np.arange(num_steps) * dt

    # Layout: for each state var -> 3 rows (pos, vel, acc), plus input rows
    input_rows = num_inputs if has_input else 0
    total_rows = num_vars * 3 + input_rows
    fig_height = max(8, total_rows * 2.8)
    fig, axes = plt.subplots(total_rows, 1, figsize=(14, fig_height), sharex=True)
    if total_rows == 1:
        axes = [axes]

    system_name = dataset.split("/")[-1]
    fig.suptitle(f"{system_name} — Training Data (sample {sample_idx})",
                 fontsize=14, fontweight="bold", y=0.998)

    row = 0
    for vi in range(num_vars):
        x = state[vi]
        x_dot = np.gradient(x, dt)
        x_ddot = np.gradient(x_dot, dt)
        vname = var_names[vi]
        color = SIGNAL_COLORS[vi % len(SIGNAL_COLORS)]

        plot_signal(axes[row], t, x,      f"{vname}\n(position)", color)
        row += 1
        plot_signal(axes[row], t, x_dot,  f"d{vname}/dt\n(velocity)", color)
        row += 1
        plot_signal(axes[row], t, x_ddot, f"d\u00b2{vname}/dt\u00b2\n(accel.)", color)
        row += 1

    for ui in range(input_rows):
        u = inp[ui]
        plot_signal(axes[row], t, u, f"u{ui+1}\n(input)", "tab:olive")
        row += 1

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    # Save with matching directory structure
    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_{sample_idx}_analysis.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

```
#### Available Tools
- `local_image_qa(image_path, question)`: Analyze the change points and dynamics of the data


### Guidelines
1. Analyze the change points and dynamics of the data from the image plot.
2. According to the change points, segment the data into different segments, these segments are saved in the a list called "SEG_DATA_LIST".
3. For each segment, fit the data using SINDy, the SINDy model is saved in the a list called "SEG_MODEL_LIST".
```python
SEG_MODEL_LIST = []
for seg_data in SEG_DATA_LIST:
    fit the data using SINDy.
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    model.fit(seg_data, t=dt)
    model.print()
    SEG_MODEL_LIST.append(model)
```
4. If the SINDy model is similar between different segments, merge the SINDy model into one.
```python
merged_model = merge_models(SEG_MODEL_LIST)
```

5. Return the SINDy model for the whole data.

### Example Usage

```python
import os
import numpy as np

#### Load the data
data = np.load(DATA_FILE_PATHS)

#### Load the image plot file
image_plot_paths = IMAGE_PLOT_PATHS


#### Ask a vision model about the data
answer = local_image_qa(
    image_path=image_plot_paths[i],
    question="What is the change points of the data?"
)
```
#### Here is the example of the SINDy model usage:
```python
__SINDY_CODE_EXAMPLE__
```

"""
        common_instruction = common_instruction.replace("__SINDY_CODE_EXAMPLE__", sindy_code_example)
        managed_tools = []
        if agent_name == "sindy_agent":
            managed_tools.append(LocalImageQATool(model_id=managed_image_tool_model_id))
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
            managed_agent_kwargs["tools"] = managed_tools
            managed_agent = CodeAgent(**managed_agent_kwargs)
            # 注入所有文件路径到 agent 状态
            managed_agent.python_executor.state["DATA_FILE_PATHS"] = sandbox_file_paths
            managed_agent.python_executor.state["DATA_FILE_PATH"] = sandbox_file_paths[0] if sandbox_file_paths else ""
        else:
            # managed agent description with all available files
            
            managed_agent_instruction = f"""## Available NPZ Files

You have access to the following NPZ data files:

{npz_files_description}
and the following image plot files:
{image_plot_description}

""" + common_instruction
            # 本地执行器：将所有数据文件路径注入到 agent 的状态中
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