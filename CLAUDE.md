# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HA-Scientist** is an agentic workflow for autonomous Hybrid Automaton (HA) system identification from time-series trajectory data. Built on the **Physics Supernova** agent architecture, it uses an iterative "Scientist-Critic" loop to refine mathematical models capturing switching dynamics.

The codebase has two main components:

1. **HA-Scientist Agent** (`run_llm_ha_gamma.py`): LLM-based iterative HA learning with visual/numerical analysis
2. **Hybrid Automaton Learning** (Dainarx_code): Traditional HA learning from trajectory data

## Architecture Overview

### Main Entry Points

**1. HA-Scientist (Primary)**
- `run_llm_ha_gamma.py` - Main LLM-based HA learning agent with iterative refinement loop

**2. Physics Problem Solving (Reference)**
- `run.py` - General physics agent (supports OpenRouter, various models)
- `run_gemini.py` - Gemini-specific physics agent

**3. Traditional HA Learning**
- `utils/Dainarx_code/main.py` - Non-LLM HA learning from trajectory data
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

*Physics Agent Tools:*
1. `WolframAlphaTool` - Mathematical computation, unit conversion
2. `AskImageTool` - Image analysis with vision models
3. `ReviewRequestTool` - Post-hoc answer checking
4. `SummarizeMemoryTool` - Memory/answer summarization for long problems

*HA-Scientist Tools* (in `TOOLNAME2TOOL` mapping):
1. `hybrid_automaton_image_analysis` → `HybridAutomatonImageTool` - Analyzes trajectory plots, can register iteration overlay images
2. `hybrid_automaton_review_expert` → `ReviewRequestTool_ha` - Reviews HA specification for correctness
3. `summarize_hybrid_automaton_iterations` → `SummarizeMemoryTool` - Summarizes iteration history
4. `validate_hybrid_automaton_specification` → `ValidateHASpecTool` - Syntax/semantic validation of HA JSON

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

### Running HA-Scientist (LLM-based HA Learning)

**Basic command:**
```bash
python run_llm_ha_gamma.py \
  --input-data-path "data_all/ATVA/ball" \
  --manager-model "gemini/gemini-flash-lite-latest" \
  --max-iterations 3
```

**Full command with all options:**
```bash
python run_llm_ha_gamma.py \
  --input-data-path "data_all/ATVA/ball" \
  --manager-model "gemini/gemini-2.5-flash-lite" \
  --manager-type CodeAgent \
  --tools-list hybrid_automaton_image_analysis validate_hybrid_automaton_specification \
  --managed-agents-list data_analysis_expert \
  --max-iterations 5 \
  --train-num 3 \
  --eval-train-num 1 \
  --target-error 0.001
```

**Key arguments:**
- `--input-data-path`: Directory containing `.npz` trace data (training: `sample_*.npz`, ground truth: `*_g/ground_truth_*.npz`)
- `--max-iterations`: Maximum refinement iterations (default: 3)
- `--train-num`: Number of training samples to show agent (default: 3)
- `--eval-train-num`: Number of ground truth files for evaluation (default: 1)
- `--feedback-top-k`: Top-k best results to include in feedback context (default: 3)
- `--target-error`: Mean difference threshold for early stopping (default: 0.001)

### Traditional HA Learning

```bash
cd utils/Dainarx_code
python main.py  # Uses automata/non_linear/duffing.json by default
```

### HA Evaluation API

```python
import sys
sys.path.insert(0, 'utils/Dainarx_code')
from HA_evaluation import HAEvaluator

evaluator = HAEvaluator(
    ha_dict=ha_specification,
    npz_file_path='data_all/ATVA/ball_g/ground_truth_0.npz',
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

### Phoenix Tracing (Optional)

Start Phoenix server for agent monitoring:
```bash
python -m phoenix.server.main serve  # Visit http://localhost:6006
```

Run with trace saving:
```bash
python run_llm_ha_gamma.py --save-traces-dir "phoenix_traces"
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
├── run_llm_ha_gamma.py                        # Main HA-Scientist entry point
├── run.py, run_gemini.py                      # Physics agent entry points
├── utils/
│   ├── utils.py                               # HAHyperparameters, ResultsAggregator, IterationResult
│   ├── prompt.py                              # HA spec examples and JSON schema documentation
│   ├── ha_spec_validator.py                   # HA specification validation
│   ├── ha_structured_output.py                # Two-stage structured output conversion
│   ├── imgTools_ha.py                         # HybridAutomatonImageTool
│   ├── reviewTools_ha.py                      # ReviewRequestTool_ha
│   ├── validateTools_ha.py                    # ValidateHASpecTool
│   ├── summemoryTools_ha.py                   # SummarizeMemoryTool
│   ├── markdown_utils.py                      # MarkdownMessage, load_trace_data_from_filepath
│   └── Dainarx_code/                          # HA learning/simulation core
│       ├── HA_evaluation.py                   # HAEvaluator class
│       ├── main.py                            # Traditional HA learning
│       ├── src/                               # Core components
│       │   ├── HybridAutomata.py              # HA simulation engine
│       │   └── ODE_System.py                  # ODE integration
│       └── automata/                          # HA JSON specifications
├── data_all/                                  # Trajectory data organized by benchmark
│   ├── ATVA/ball/, ATVA/ball_g/               # Training data + ground truth
│   └── FaMoS/*, non_linear/*                  # Other benchmarks
├── evaluation_results/                        # Output directory
│   └── <benchmark>/<system>/runs/<run_id>/    # Per-run artifacts
│       ├── iter_1/, iter_2/, ...              # Per-iteration results
│       ├── best_iter_N/                       # Copy of best iteration
│       └── best_ha_specification.json         # Final best HA spec
└── task_prompts/                              # Generated task prompts per iteration
```

## HA-Scientist Iterative Loop

The main workflow (`run_llm_ha_gamma.py`) implements an iterative refinement loop:

1. **Initial Prompt**: Agent receives trajectory plots, NPZ data access via managed agents, and HA JSON template
2. **Hypothesis Generation**: Agent proposes HA specification (modes, ODEs, guards, resets)
3. **Evaluation**: `HAEvaluator` simulates HA and computes metrics (mean_diff, max_diff, TC)
4. **Feedback**: LLM summarizes errors + overlay plots registered for next iteration
5. **Refinement**: Top-k best results fed back as context for next iteration

**Key components:**
- `ResultsAggregator` (`utils/utils.py`): Tracks iteration results, selects top-k diverse feedback
- `HAHyperparameters`: Dataclass storing all experiment configuration
- `evaluate_ha_specification_with_feedback()`: Evaluation + artifact generation + LLM summary

**Early stopping conditions:**
- Target error achieved (`--target-error`, default: 0.001)
- No improvement for N iterations (`--no-improvement-patience`, default: 10)
- Near-perfect fit (error < 0.0001)

## Important Caveats

1. **Agent Step Limits**: Agents default to `max_steps=80`. Long problems may hit this limit.

2. **Image Resolution**: Tools receive compressed images (~1080px) in initial prompt, but can access high-res via `worker_agent.markdown_content_high_res_image`.

3. **Input Time Indexing**: In HA evaluation, `input_array[i]` corresponds to time `(i+1)*dt`, not `i*dt`. The `analyticalInput()` function handles this automatically.

4. **Mode Numbering**: HA modes are 1-indexed (not 0-indexed). Ground truth mode IDs should match learned mode IDs after bipartite matching.

5. **Tool Injection Timing**: Always inject `worker_agent` reference **after** agent creation but **before** calling `agent.run()`.

6. **API Key Precedence**: Scripts load from `.env` file. Ensure correct API base is set for the model provider being used.

## Data Organization

**Training data** (`data_all/<benchmark>/<system>/`):
- `sample_0.npz`, `sample_1.npz`, ... - Trajectory samples for agent to analyze
- Corresponding plots generated and shown to agent

**Ground truth** (`data_all/<benchmark>/<system>_g/`):
- `ground_truth_0.npz`, `ground_truth_1.npz`, ... - Validation trajectories
- Used by `HAEvaluator` to compute metrics

## Related Documentation

- `README.md` - User-facing documentation
- `utils/Dainarx_code/HA_evaluation_README.md` - Detailed HA evaluation API
- `utils/Dainarx_code/automata/json_readme.md` - HA JSON format specification
- Paper: [Physics Supernova ArXiv](https://arxiv.org/abs/2509.01659)

## Guideline 
1. When modifying output formatting or display strings, show the exact expected output format before making changes. Ask for clarification if the desired format isn't explicit.
2. For complex multi-step implementations, use @planning-with-files