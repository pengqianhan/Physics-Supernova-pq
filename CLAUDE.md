# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HA-Scientist** (Hybrid Automaton System Identification Agent) - an agentic workflow for autonomously identifying Hybrid Automaton (HA) models from time-series trajectory data using an iterative "Scientist-Critic" loop with visual and numerical fusion.

## Architecture Overview

### Main Entry Point

**`run_pure_python_ha.py`** - Pure Python class-based HA learning (current approach)
- Generates Python classes inheriting from `HybridAutomatonBase`
- Uses `CodeAgent` with LLM (Gemini) to iteratively refine models
- Auto-detects data dimensions from `.npz` files
- Includes parameter optimization via simulated annealing
- Early stopping based on error thresholds and patience

### Core Components

```
utils/
├── ha_base_class.py          # HybridAutomatonBase - abstract base for all HA classes
├── ha_class_template.py      # Templates for LLM-generated HA classes
├── ha_class_simulator.py     # Simulates Python class HAs (PythonClassHAEvaluator)
├── ha_class_optimizer.py     # Parameter optimization (optimize_ha_params)
├── ha_class_validator.py     # Syntax validation (validate_python_class_syntax)
├── evaluate_pure_python_ha.py # End-to-end evaluation pipeline
├── pure_python_workflow.py   # Task prompt generation (generate_pure_python_task)
├── llm_evaluator.py          # LLM critique generation
├── imgTools_ha.py            # HybridAutomatonImageTool - vision-based trajectory analysis
└── validateTools_ha.py       # ValidateHASpecTool - JSON Schema validation (legacy)
```

### HybridAutomatonBase Contract

All generated HA classes must inherit from `HybridAutomatonBase` and implement:

```python
class MyHA(HybridAutomatonBase):
    def __init__(self):
        self.params = [...]      # Max 10 tunable parameters
        self.var = "x1"          # State variables (comma-separated)
        self.input = "u1"        # Input variables (comma-separated)
        self.dt = 0.001          # Time step
        self.total_time = 10.0   # Simulation duration
        self.order = 2           # ODE order (1 or 2)

    def num_modes(self) -> int: ...
    def mode_dynamics(self, mode_id, x, u) -> str: ...     # Returns ODE string
    def guard_condition(self, src, tgt, x, u) -> bool: ... # Transition condition
    def reset_map(self, src, tgt, x, u) -> None: ...       # State reset on transition
```

### ODE String Format

- `x[0]` = variable value, `x[1]` = first derivative, `x[k]` = k-th derivative
- Left side must be highest derivative
- Example (2nd-order): `"x1[2] = -0.5*x1[1] - 5*x1[0] + u['u1']"`

## Common Development Commands

### Environment Setup

```bash
pip install -r requirements.txt

# For Phoenix telemetry (optional)
pip install arize-phoenix openinference-instrumentation-smolagents
```

### API Key Configuration

Create `.env` file:
```bash
GEMINI_API_KEY=...    # Required for LLM agent
```

### Running HA-Scientist

```bash
# Basic usage
python run_pure_python_ha.py \
  --input-data-path data_all/non_linear/duffing \
  --manager-model gemini/gemini-flash-lite-latest \
  --max-iterations 3 \
  --optimization-iters 50

# With specific optimizer
python run_pure_python_ha.py \
  --input-data-path data_all/ATVA/ball \
  --manager-model gemini/gemini-flash-lite-latest \
  --optimizer-type simulated_annealing \
  --target-error 0.01
```

**Key Arguments:**
- `--input-data-path`: Directory with `.npz` trace data files
- `--manager-model`: LLM model ID (e.g., `gemini/gemini-flash-lite-latest`)
- `--max-iterations`: Maximum refinement iterations (default: 3)
- `--optimization-iters`: Parameter optimization iterations (default: 50)
- `--optimizer-type`: `simulated_annealing` or `hill_climbing`
- `--target-error`: Early stop threshold (default: 0.01)
- `--tools-list`: Tools available to agent (default: `hybrid_automaton_image_analysis`, `validate_hybrid_automaton_specification`)

**Output:**
- `evaluation_results/iter_N/` - Per-iteration plots and metrics
- `evaluation_results/best/best_ha_class.py` - Best HA class code
- `evaluation_results/best/best_params.npy` - Optimized parameters
- `task_prompts/iter_N/task.md` - Generated task prompts

### Testing

```bash
# Run workflow tests
python tests/test_pure_python_workflow.py

# Test template generation standalone
python utils/ha_class_template.py

# Test evaluation pipeline
python utils/evaluate_pure_python_ha.py
```

### Traditional HA Learning (Legacy)

```bash
cd utils/Dainarx_code
python main.py  # Classical clustering + SVM guards
```

## Data Formats

### NPZ Trajectory Data

```python
{
  'state': np.ndarray,         # Shape: (num_states, num_steps)
  'input': np.ndarray,         # Shape: (num_inputs, num_steps) or (num_steps,)
  'mode': np.ndarray,          # Shape: (num_steps,) - mode ID at each step
  'change_points': np.ndarray  # Shape: (num_transitions,) - indices of mode switches
}
```

### HA JSON Format (Legacy)

See `utils/Dainarx_code/automata/json_readme.md` for the JSON schema used by the traditional HA learning pipeline.

## Key Design Patterns

### Agent Tool Injection

```python
agent = create_agent(model_id, tools_list)
# Tools get worker_agent reference for context access
for tool in agent.tools.values():
    if hasattr(tool, 'worker_agent'):
        tool.worker_agent = agent
```

### ResultsAggregator Pattern

```python
aggregator = ResultsAggregator(top_k=3, min_gap=0.005)
aggregator.add_result(IterationResult(...))

# Early stopping check
should_stop, reason = aggregator.should_early_stop(
    target_error=0.01, min_iterations=2, no_improvement_patience=3
)
```

### Class Execution Namespace

```python
from utils.ha_base_class import HybridAutomatonBase

namespace = {'HybridAutomatonBase': HybridAutomatonBase}
exec(class_code, namespace)
ha_instance = namespace['HybridAutomaton']()
```

## Common Pitfalls

1. **Variable Count Mismatch**: LLMs may incorrectly convert higher-order ODEs to state-space form. The template pre-fills `var` and `input` to prevent this.

2. **Mode Numbering**: HA modes are 1-indexed (not 0-indexed). Mode IDs must be `>= 1`.

3. **Input Time Indexing**: In HA evaluation, `input_array[i]` corresponds to time `(i+1)*dt`, not `i*dt`.

4. **ODE Order**: Single-variable systems are typically 2nd order (oscillators), multi-variable systems are typically 1st order.

5. **Agent Step Limits**: Agents default to `max_steps=100`. Increase for complex tasks.

6. **Params Array**: Maximum 10 parameters in `self.params`. Use indices consistently across all methods.

## Telemetry (Optional)

Phoenix monitoring is auto-enabled if installed:
```bash
pip install arize-phoenix openinference-instrumentation-smolagents
# Visit http://localhost:6006 during runs
```
