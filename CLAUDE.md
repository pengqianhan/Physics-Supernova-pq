# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository combines **Physics Supernova** (an AI agent for physics problems) with **Hybrid Automaton Learning** capabilities. The codebase has two main components:

1. **Physics Supernova Agent**: AI system for solving IPhO-level physics problems using multi-agent architecture with specialized tools
2. **Hybrid Automaton Learning** (Dainarx_code): System for learning hybrid automaton specifications from trajectory data

## Architecture Overview

### Two Main Entry Points

**1. Physics Problem Solving**
- `run.py` - General physics agent (supports OpenRouter, various models)
- `run_gemini.py` - Gemini-specific physics agent
- `run_llm_ha_alpha.py` - LLM agent for hybrid automaton learning/improvement

**2. Hybrid Automaton Learning**
- `utils/Dainarx_code/main.py` - Traditional HA learning from trajectory data
- `utils/Dainarx_code/HA_evaluation.py` - Evaluation framework for HA specifications

### Agent System Architecture

The project uses **smolagents** framework (HuggingFace) with two agent types:
- **CodeAgent**: Can execute Python code, use tools, and manage sub-agents
- **ToolCallingAgent**: Function-calling style agent (no code execution)

Agents are configured with:
- **Manager Agent**: Main reasoning/planning agent
- **Specialized Tools**: Wolfram, Image Analysis, Review, Memory Summarization
- **Managed Agents** (optional): Subordinate agents for specific tasks

### Tool System

Tools inherit from `smolagents.default_tools.Tool` and follow delayed injection pattern:

```python
# In agent creation (run.py:106-107, run_gemini.py:113-114)
for toolName in managerAgent.tools:
    managerAgent.tools[toolName].worker_agent = managerAgent
```

Tools can access parent agent via `self.worker_agent` to retrieve:
- `worker_agent.markdown_content_high_res_image` - Full-resolution images
- Other agent state/context as needed

**Available Tools:**
1. `WolframAlphaTool` - Mathematical computation, unit conversion
2. `AskImageTool` / `HybridAutomatonImageTool` - Image analysis with vision models
3. `ReviewRequestTool` / `ReviewRequestTool_ha` - Post-hoc answer checking
4. `SummarizeMemoryTool` - Memory/answer summarization for long problems

### Hybrid Automaton System

**Core Components** (in `utils/Dainarx_code/src/`):
- `HybridAutomata.py` - HA simulation engine
- `ODE_System.py` - ODE integration and input handling
- `BuildSystem.py` - Constructs HA from learned components
- `GuardLearning.py` - Learns transition guards using SVM
- `Clustering.py` - Clusters trajectory slices into modes
- `Evaluation.py` - Computes evaluation metrics

**Pipeline Flow**:
1. Load trajectory data from `.npz` files
2. Detect change points (mode switches)
3. Slice trajectories at change points
4. Cluster slices into modes
5. Learn ODEs for each mode
6. Learn guards between modes
7. Build complete HA specification
8. Simulate and evaluate

### Input Data Flow in HA System

**Critical Detail**: Input data in HA evaluation uses time-indexed array functions:

```python
# Array-to-function conversion (ODE_System.py:38-89)
def analyticalInput(input_array):
    # Creates closure: idx = int(round(t / dt - 1))
    # input_array[i] corresponds to time (i+1)*dt
```

When evaluating HA:
1. NPZ file contains full input time series
2. `_prepare_initial_state()` passes entire array to HA
3. `analyticalInput()` wraps array in time-indexed function
4. During simulation, `getInput(t)` retrieves value at time `t`

## Common Development Commands

### Environment Setup

```bash
# Install dependencies
python -m pip install -U pip
pip install tenacity
pip install smolagents
pip install smolagents[litellm]
pip install loguru
pip install python-dotenv

# Required for HA learning
pip install numpy scipy matplotlib
```

### API Key Configuration

Create `.env` file in project root:

```bash
OPENROUTER_API_KEY=sk-...         # For Physics Supernova (OpenRouter)
GEMINI_API_KEY=...                # For Gemini-specific scripts
WOLFRAM_APP_ID=...                # For WolframAlpha tool
HF_TOKEN=hf_...                   # If using HF models
```

### Running Physics Agent

**Single problem:**
```bash
python run.py \
  --input-markdown-file examples/Problems/example/example1problem.md \
  --manager-model openrouter/google/gemini-2.5-pro \
  --manager-type CodeAgent \
  --tools-list wolfram_alpha_query ask_image_expert ask_review_expert finalize_part_answer \
  --image-tool-model openrouter/google/gemini-2.5-pro \
  --review-tool-model openrouter/google/gemini-2.5-pro
```

**Batch execution:**
```bash
# Edit run_scripts/batchrun.py to configure MAX_THREADS and ARGS_LIST
python run_scripts/batchrun.py

# IPhO problems
python run_scripts/batchrun_IPhO.py

# Wolfram QA tasks
python run_scripts/batchrun_wolftask.py
```

### Running HA Learning

**Traditional HA learning:**
```bash
cd utils/Dainarx_code
python main.py  # Uses automata/non_linear/duffing.json by default
```

**LLM-based HA learning:**
```bash
python run_llm_ha_alpha.py
```

**Evaluate HA specification:**
```python
import sys
sys.path.insert(0, 'utils/Dainarx_code')
from HA_evaluation import HAEvaluator

evaluator = HAEvaluator(
    ha_dict=ha_specification,
    npz_file_path='utils/Dainarx_code/data_duffing/test_data0.npz',
    dt=0.001,
    total_time=10.0
)

# Run evaluation with plot and metrics
results = evaluator(
    plot_mode='overlay',
    save_path='output_comparison.png',
    show_plot=False
)
```

### Testing

```bash
# Test image tool
python test_imgTools_ha.py

# Test review tool
python test_reviewTools_ha.py

# Test API connectivity
python test_api.py

# Run HA evaluation tests
cd utils/Dainarx_code
python HA_evaluation.py
```

## Key Data Formats

### Markdown Problem Format

Physics problems are stored as markdown with embedded images:

```markdown
# Problem Statement

![Figure 1](path/to/image1.png)

Question text here. Reference images as <image_1>, <image_2>.
```

Images are:
- **High-res**: Stored in `agent.markdown_content_high_res_image` (MarkdownMessage object)
- **Compressed**: Passed to agent via `images` parameter (max 1080px)

### Hybrid Automaton JSON Format

**Structure** (see `utils/Dainarx_code/automata/json_readme.md`):

```json
{
  "automaton": {
    "var": "x1, x2",           // State variables (comma-separated)
    "input": "u1",             // Input variables (comma-separated)
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -0.1*x2[0] - x1[0] - x1[0]**3 + u1"
        // ODEs: x[k] is k-th derivative, x[0] is value
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "x1 >= 5",
        "reset": {              // Optional reset map
          "x1": ["", "x1"],     // ["constant", "linear_term"]
          "x2": ["", "x2"]
        }
      }
    ]
  },
  "config": {
    "dt": 0.001,               // Time step
    "total_time": 10.0,        // Simulation duration
    "dim": 2,                  // ODE order for learning
    "other_items": ""          // Nonlinear terms for learning
  }
}
```

**Important conventions:**
- `x[0]` = variable value (not derivative)
- `x[1]` = first derivative
- `x[k]` = k-th derivative
- Left side of ODE must be highest-order derivative
- All variables in `var` must have an ODE

### NPZ Trajectory Data Format

Ground truth trajectories in `.npz` format:

```python
{
  'state': np.ndarray,         # Shape: (num_states, num_steps)
  'input': np.ndarray,         # Shape: (num_inputs, num_steps) or (num_steps,)
  'mode': np.ndarray,          # Shape: (num_steps,) - mode ID at each step
  'change_points': np.ndarray  # Shape: (num_transitions,) - indices of mode switches
}
```

## Important Code Patterns

### Agent Creation Pattern

```python
# 1. Load markdown content with high-res images
markdown_content = load_markdown_from_filepath(input_file)

# 2. Create agent with tools
agent = create_agent(
    model_id="...",
    input_markdown_file=input_file,
    tools_list=["wolfram_alpha_query", "ask_image_expert", ...],
    manager_type="CodeAgent"
)

# 3. Inject agent reference into tools (delayed injection)
for toolName in agent.tools:
    agent.tools[toolName].worker_agent = agent

# 4. Store high-res images in agent
agent.markdown_content_high_res_image = markdown_content

# 5. Prepare task with compressed images
task, compressed_images = obtain_task_and_images(input_file, tools_list)

# 6. Run agent
agent.run(task, images=compressed_images)
```

### Tool Implementation Pattern

```python
from smolagents import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "..."
    inputs = {"param": {"type": "string", "description": "..."}}
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.worker_agent = None  # Will be injected later

    def forward(self, param: str) -> str:
        # Access agent context
        high_res_images = self.worker_agent.markdown_content_high_res_image

        # Tool logic here
        return result
```

### HA Evaluation Pattern

```python
from utils.Dainarx_code.HA_evaluation import HAEvaluator

# Create evaluator
evaluator = HAEvaluator(ha_dict, npz_file_path, dt=0.001)

# Option 1: One-line evaluation
results = evaluator()  # Shows plot and prints metrics

# Option 2: Step-by-step
gt_data = evaluator.load_ground_truth()
sim_data = evaluator.simulate()
metrics = evaluator.compute_metrics()
evaluator.plot(plot_mode='overlay', save_path='output.png')

# Option 3: Programmatic (batch processing)
results = evaluator(
    plot_mode='overlay',
    save_path='output.png',
    show_plot=False,
    print_metrics=False
)
```

## Model Configuration Notes

**API Bases:**
- OpenRouter (default): `https://openrouter.ai/api/v1`
- Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`

**Model Selection:**
- Use `openrouter/google/gemini-2.5-pro` for complex reasoning
- Use `gemini/gemini-2.5-flash-lite` for faster/cheaper operations
- Vision models needed for `ask_image_expert` tool

**LiteLLM Configuration:**
```python
model = LiteLLMModel(
    model_id="gemini/gemini-2.5-flash-lite",
    api_key=os.environ.get("GEMINI_API_KEY"),
    max_completion_tokens=24576,
    num_retries=3,
    timeout=1200
)
```

## Directory Structure Key Points

```
Physics-Supernova-pq/
├── run.py, run_gemini.py, run_llm_ha_alpha.py  # Main entry points
├── utils/
│   ├── imgTools.py, imgTools_ha.py             # Image analysis tools
│   ├── reviewTools.py, reviewTools_ha.py       # Review tools
│   ├── wolframTools.py, summemoryTools.py      # Other tools
│   ├── markdown_utils.py                       # MarkdownMessage handling
│   └── Dainarx_code/                           # HA learning system
│       ├── main.py                             # Traditional HA learning
│       ├── HA_evaluation.py                    # Evaluation framework
│       ├── CreatData.py                        # Generate synthetic data
│       ├── src/                                # Core HA components
│       │   ├── HybridAutomata.py              # HA simulation
│       │   ├── ODE_System.py                  # ODE integration
│       │   ├── BuildSystem.py, GuardLearning.py
│       │   └── Clustering.py, ChangePoints.py
│       ├── automata/                           # HA specifications
│       │   ├── linear/, non_linear/, ATVA/, FaMoS/
│       │   └── json_readme.md                 # Format documentation
│       └── data_*/                            # Generated trajectory data
├── run_scripts/                               # Batch execution scripts
│   ├── batchrun.py, batchrun_IPhO.py
│   └── batchrun_wolftask.py
├── examples/Problems/                         # Physics problems
├── prompts_ha/                                # Prompts for HA learning
└── learning_notes/                            # Documentation/examples
```

## LLM-Based HA Learning (LLM-LEx)

This project implements LLM-based hybrid automaton learning using iterative refinement:

**Method** (from `.cursor/rules/method.mdc`):
- Uses experience buffer with island-based populations
- Generates HA hypotheses via LLM prompting
- Evaluates against trajectory data
- Samples high-quality examples for in-context learning
- Periodically resets worst-performing islands

**Key hyperparameters:**
- `b = 4` equation programs per generation
- `e = 4` parallel evaluators
- `m = 10` islands for diversity
- `τ = 0.8` generation temperature
- Max 10 parameters per equation
- 30s timeout, 2GB memory limit per evaluation

## Important Caveats

1. **Agent Step Limits**: Agents default to `max_steps=80`. Long problems may hit this limit.

2. **Image Resolution**: Tools receive compressed images (~1080px) in initial prompt, but can access high-res via `worker_agent.markdown_content_high_res_image`.

3. **Input Time Indexing**: In HA evaluation, `input_array[i]` corresponds to time `(i+1)*dt`, not `i*dt`. The `analyticalInput()` function handles this automatically.

4. **Mode Numbering**: HA modes are 1-indexed (not 0-indexed). Ground truth mode IDs should match learned mode IDs after bipartite matching.

5. **Tool Injection Timing**: Always inject `worker_agent` reference **after** agent creation but **before** calling `agent.run()`.

6. **API Key Precedence**: Scripts load from `.env` file. Ensure correct API base is set for the model provider being used.

## Related Documentation

- `README.md` - User-facing documentation, IPhO results
- `utils/Dainarx_code/HA_evaluation_README.md` - Detailed HA evaluation API
- `utils/Dainarx_code/automata/json_readme.md` - HA JSON format specification
- `.cursor/rules/method.mdc` - LLM-SR algorithm details
- Paper: [Physics Supernova ArXiv](https://arxiv.org/abs/2509.01659)
