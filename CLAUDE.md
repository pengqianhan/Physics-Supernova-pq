# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **HA-Scientist** (Hybrid Automaton System Identification Agent), an agentic workflow for autonomously identifying Hybrid Automaton (HA) models from time-series trajectory data. The codebase evolved from **Physics Supernova** architecture and has two main components:

1. **HA-Scientist Agent** (`run_llm_ha_beta.py`): Iterative "Scientist-Critic" loop for HA model identification with visual and numerical fusion
2. **Traditional HA Learning** (Dainarx_code): Classical system identification using clustering, SVM-based guard learning, and ODE fitting
3. **Physics Supernova Agent** (legacy): AI system for solving IPhO-level physics problems

## Architecture Overview

### Main Entry Points

**1. HA-Scientist Iterative Loop (Primary)**
- `run_llm_ha_beta.py` - **Main entry point** for iterative HA model identification
  - Implements "Scientist-Critic" loop with feedback aggregation
  - Uses ResultsAggregator for intelligent top-k feedback selection
  - Includes early stopping based on error thresholds and patience
  - Auto-detects data dimensions from `.npz` files
  - Generates dynamic HA templates with pre-filled variables

**2. Traditional HA Learning**
- `utils/Dainarx_code/main.py` - Classical HA learning from trajectory data
- `utils/Dainarx_code/HA_evaluation.py` - Evaluation framework for HA specifications

**3. Physics Problem Solving (Legacy)**
- `run.py` - General physics agent (supports OpenRouter, various models)
- `run_gemini.py` - Gemini-specific physics agent
- `run_llm_ha_alpha.py` - Early LLM-based HA learning (superseded by beta)

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

*For HA Learning:*
1. `HybridAutomatonImageTool` (`utils/imgTools_ha.py`) - Vision-based trajectory analysis
   - Analyzes trajectory plots to detect mode switches, oscillations, discontinuities
   - Uses vision models (Gemini) for qualitative system behavior insights
2. `ValidateHASpecTool` (`utils/validateTools_ha.py`) - JSON Schema validation
   - Validates HA specification syntax and semantics
   - Auto-fixes common errors before submission
3. `SummarizeMemoryTool` (`utils/summemoryTools_ha.py`) - Iteration summarization
   - Summarizes previous iterations for context management
4. `ReviewRequestTool_ha` (`utils/reviewTools_ha.py`) - HA spec review
   - Post-hoc checking of HA specifications for completeness

*For Physics Problems (Legacy):*
1. `WolframAlphaTool` - Mathematical computation, unit conversion
2. `AskImageTool` - Image analysis with vision models
3. `ReviewRequestTool` - Post-hoc answer checking

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
# Install dependencies (use requirements.txt)
python -m pip install -U pip
pip install -r requirements.txt

# Or install manually
pip install tenacity smolagents[litellm] loguru python-dotenv
pip install numpy scipy matplotlib pandas seaborn scikit-learn networkx

# Optional: E2B sandbox environment
pip install smolagents[e2b]
```

### API Key Configuration

Create `.env` file in project root:

```bash
OPENROUTER_API_KEY=sk-...         # For Physics Supernova (OpenRouter)
GEMINI_API_KEY=...                # For Gemini-specific scripts
WOLFRAM_APP_ID=...                # For WolframAlpha tool
HF_TOKEN=hf_...                   # If using HF models
```

### Running HA-Scientist (Primary Use Case)

**Basic usage:**
```bash
python run_llm_ha_beta.py \
  --input-data-path data_all/non_linear/duffing \
  --manager-model gemini/gemini-flash-lite-latest \
  --manager-type CodeAgent \
  --max-iterations 3 \
  --tools-list hybrid_automaton_image_analysis summarize_hybrid_automaton_iterations validate_hybrid_automaton_specification
```

**With managed agents for data analysis:**
```bash
python run_llm_ha_beta.py \
  --input-data-path data_all/non_linear/duffing \
  --manager-model gemini/gemini-flash-lite-latest \
  --managed-agents-list data_analysis_expert \
  --managed-agents-list-model gemini/gemini-flash-lite-latest \
  --max-iterations 5 \
  --target-error 0.01
```

**Key Arguments:**
- `--input-data-path`: Directory with `.npz` trace data files
- `--manager-model`: LLM for main agent (e.g., `gemini/gemini-flash-lite-latest`)
- `--manager-type`: `CodeAgent` (can execute Python) or `ToolCallingAgent`
- `--max-iterations`: Maximum refinement iterations (default: 3)
- `--target-error`: Early stop threshold (default: 0.01)
- `--feedback-top-k`: Number of top specs in feedback (default: 3)
- `--managed-agents-list`: Sub-agents like `data_analysis_expert`

**Output:**
- `evaluation_results/iter_N/ha_eval_*.png` - Trajectory comparison plots
- `evaluation_results/iter_N/ha_eval_*.txt` - Metrics and HA spec
- `evaluation_results/best_ha_specification.json` - Best result across iterations
- `task_prompts/iter_N_task.md` - Generated task prompts

### Running Physics Agent (Legacy)

**Single problem:**
```bash
python run.py \
  --input-markdown-file examples/Problems/example/example1problem.md \
  --manager-model openrouter/google/gemini-2.5-pro \
  --manager-type CodeAgent \
  --tools-list wolfram_alpha_query ask_image_expert ask_review_expert finalize_part_answer
```

### Running Traditional HA Learning

**Classical HA learning (clustering + SVM guards):**
```bash
cd utils/Dainarx_code
python main.py  # Uses automata/non_linear/duffing.json by default
```

**Evaluate any HA specification:**
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

## HA-Scientist Iterative Loop Architecture

The main workflow (`run_llm_ha_beta.py`) implements an iterative refinement pattern:

### Core Components

**1. ResultsAggregator** (`utils/utils.py`):
- Tracks all iteration results with metrics
- Selects top-k diverse specifications for feedback (diversity via min_gap filtering)
- Implements early stopping logic:
  - Target error threshold reached
  - No improvement for N iterations (patience)
  - Near-perfect fit achieved (error < 0.0001)
- Provides dynamic error thresholds

**2. Feedback Generation** (`obtain_task_and_images()`):
- First iteration: No feedback (cold start)
- Later iterations:
  - Top-k diverse specs ranked by error
  - Most recent iteration feedback
  - Metrics from previous evaluations (TC, max_diff, mean_diff)

**3. HA Specification Validation** (`utils/ha_spec_validator.py`):
- Extracts JSON from agent output (handles markdown code blocks)
- Validates against JSON Schema
- Auto-fixes common errors (missing fields, type mismatches)
- Returns preprocessed spec for evaluation

**4. Evaluation with Feedback** (`evaluate_ha_specification_with_feedback()`):
- Simulates HA against ground truth `.npz` data
- Computes metrics: TC (change-point error), max_diff, mean_diff, RMSE
- Generates plots: overlay (HA vs GT), separate subplots
- Returns structured feedback for next iteration

### Key Design Patterns

**Dynamic Template Generation**:
```python
# Auto-detects num_variables and num_inputs from .npz file
num_variables, num_inputs = get_data_dimensions(input_data_path)
# Generates HA template with pre-filled var and input fields
dynamic_ha_template = generate_dynamic_ha_template(num_variables, num_inputs)
```

**Managed Agent State Injection**:
```python
# NPZ file paths are injected into managed agent state
managed_agent.python_executor.state["DATA_FILE_PATHS"] = npz_paths_list
managed_agent.python_executor.state["DATA_FILE_PATH"] = npz_paths_list[0]
```

**Top-K Diverse Feedback Selection**:
```python
# From utils/utils.py ResultsAggregator.get_top_k_feedback()
# Sorts by error, then filters by minimum gap for diversity
for result in sorted_results:
    if result.error_value - last_accepted_error >= self.min_gap:
        distinct_results.append(result)
```

## Important Caveats and Common Pitfalls

### HA-Scientist Specific

1. **Variable Count Mismatch**: The most common error is LLMs incorrectly converting higher-order ODEs to state-space form.
   - Ground truth has 1 variable → Use `x1[2] = ...` (2nd-order ODE)
   - LLM incorrectly outputs 2 variables → `x1[1] = x2[0], x2[1] = ...` ❌
   - Fix: Pre-filled `var` and `input` fields in dynamic template prevent this

2. **Input Time Indexing**: In HA evaluation, `input_array[i]` corresponds to time `(i+1)*dt`, not `i*dt`. The `analyticalInput()` function in `ODE_System.py:38-89` handles this automatically.

3. **Mode Numbering**: HA modes are 1-indexed (not 0-indexed). Mode IDs must be `>= 1`.

4. **Feedback Loop Design**:
   - `feedback_top_k`: Too high → context overflow; too low → insufficient examples
   - `feedback_min_gap`: Too small → redundant specs; too large → skip good examples
   - Recommended: `top_k=3`, `min_gap=0.005` for most systems

5. **Early Stopping**: The loop stops early if:
   - Best error < `target_error` (default 0.01)
   - No improvement for `no_improvement_patience` iterations (default 3)
   - Near-perfect fit: error < 0.0001

### General Agent Issues

6. **Agent Step Limits**: Agents default to `max_steps=80`. Increase for complex tasks.

7. **Image Resolution**: Tools receive compressed images (~1080px) in initial prompt, but can access high-res via `worker_agent.markdown_content_high_res_image`.

8. **Tool Injection Timing**: Always inject `worker_agent` reference **after** agent creation but **before** calling `agent.run()`:
   ```python
   for toolName in agent.tools:
       agent.tools[toolName].worker_agent = agent
   ```

9. **API Key Precedence**: Scripts load from `.env` file. Ensure correct API base is set:
   - Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`
   - OpenRouter: `https://openrouter.ai/api/v1`

10. **Managed Agent File Access**:
    - E2B sandbox: Files must be uploaded to sandbox first
    - Local executor: Use absolute paths or inject via `state` dict

## Related Documentation

**HA-Scientist:**
- `README.md` - Project overview and usage instructions
- `utils/Dainarx_code/HA_evaluation_README.md` - Detailed HA evaluation API
- `utils/Dainarx_code/automata/json_readme.md` - HA JSON format specification
- `prompts_ha/prompts.py` - HA specification documentation and JSON Schema
- `task_prompts/iter_N_task.md` - Generated task prompts (auto-created during runs)

**Physics Supernova (Legacy):**
- Paper: [Physics Supernova ArXiv](https://arxiv.org/abs/2509.01659)
- `GEMINI.md` - Gemini-specific configuration notes

**Learning Notes:**
- `learning_notes/` - Internal documentation and examples
- `utils/Dainarx_code/error_analysis.md` - Error analysis for HA learning
