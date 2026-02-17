# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HA-Scientist** is an agentic workflow for autonomous Hybrid Automaton (HA) system identification from time-series trajectory data. Built on the **Physics Supernova** agent architecture, it uses an iterative "Scientist-Critic" loop to refine mathematical models capturing switching dynamics.

Two main components:
1. **HA-Scientist Agent** (`run_llm_ha_gamma.py`): LLM-based iterative HA learning with visual/numerical analysis
2. **Hybrid Automaton Learning** (`utils/Dainarx_code/`): Traditional HA learning from trajectory data

## Architecture

### Core Pipeline: HA-Scientist Iterative Loop

```
LLM proposes HA spec → HAEvaluator simulates → metrics computed → feedback → LLM refines
                         ↕
              Two-stage structured output (ha_structured_output.py):
              Agent free-text → LLM JSON extraction → validated HA dict
```

1. Agent receives trajectory plots + NPZ data access + HA JSON template
2. Agent proposes HA specification (modes, ODEs, guards, resets)
3. Agent output converted to structured JSON via `convert_agent_result_to_ha()` (two-stage: free-text → LLM extraction → validation)
4. `HAEvaluator` simulates HA and computes metrics (mean_diff, max_diff, TC)
5. LLM summarizes errors + overlay plots registered for next iteration
6. Top-k best results (via `ResultsAggregator`) fed back as context

Early stopping: target error achieved, no improvement for N iterations, or near-perfect fit (< 0.0001).

### Agent Framework

Uses **smolagents** (HuggingFace) with **CodeAgent** (can execute code + use tools) or **ToolCallingAgent** (function-calling only).

**Critical pattern — delayed tool injection:**
```python
for toolName in managerAgent.tools:
    managerAgent.tools[toolName].worker_agent = managerAgent
```
Tools access parent agent via `self.worker_agent` (e.g., to get high-res images). Must inject **after** agent creation but **before** `agent.run()`.

### HA-Scientist Tools (in `TOOLNAME2TOOL` mapping)

| Tool name | Class | Purpose |
|---|---|---|
| `hybrid_automaton_image_analysis` | `HybridAutomatonImageTool` | Analyze trajectory plots, register overlay images |
| `hybrid_automaton_review_expert` | `ReviewRequestTool_ha` | Review HA spec for correctness |
| `summarize_hybrid_automaton_iterations` | `SummarizeMemoryTool` | Summarize iteration history |
| `validate_hybrid_automaton_specification` | `ValidateHASpecTool` | Syntax/semantic validation of HA JSON |

### Parameter Optimization Pipeline (Active Development)

Two complementary optimizers exist for post-LLM coefficient refinement:

1. **`optimize_ha_params.py`** — Nevergrad-based: LLM extracts tunable parameters via Pydantic schema → Nevergrad gradient-free optimization → HAEvaluator scoring. Uses `loky.ProcessPoolExecutor` for parallel evaluation.

2. **`optimize_sindy_nevergrad.py`** — Combined SINDy + Nevergrad: PySINDy fits mode ODEs from trajectory segments, then Nevergrad optimizes edge parameters (guards + resets).

The current development direction (`improve_plan.md`) is a **skeleton-first approach**: LLM identifies ODE structure only (no coefficients), then numerical fitting estimates parameters segment-by-segment. This separates structural identification (LLM strength) from parameter estimation (optimizer strength).

### HA Simulation Engine (`utils/Dainarx_code/src/`)

- `HybridAutomata.py` — HA simulation engine
- `ODE_System.py` — ODE integration; `analyticalInput()` wraps input arrays as time-indexed functions
- `BuildSystem.py` — Constructs HA from learned components
- `GuardLearning.py` — Learns transition guards using SVM
- `Clustering.py` — Clusters trajectory slices into modes
- `Evaluation.py` — Computes evaluation metrics

### Key Classes (`utils/utils.py`)

- `HAHyperparameters` — Dataclass storing all experiment configuration
- `IterationResult` — Single iteration's HA spec + metrics
- `ResultsAggregator` — Tracks iteration results, selects top-k diverse feedback for next iteration

## Common Development Commands

### Environment Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# Additional dependencies for optimization pipeline
pip install nevergrad loky pydantic pysindy
```

API keys in `.env`:
```bash
GEMINI_API_KEY=...                # For Gemini models (primary)
OPENROUTER_API_KEY=sk-...         # For OpenRouter models
WOLFRAM_APP_ID=...                # For WolframAlpha tool
```

### Running HA-Scientist

```bash
# Basic run
python run_llm_ha_gamma.py \
  --input-data-path "data_all/ATVA/ball" \
  --manager-model "gemini/gemini-2.5-flash-lite" \
  --max-iterations 3

# Full options
python run_llm_ha_gamma.py \
  --input-data-path "data_all/ATVA/ball" \
  --manager-model "gemini/gemini-2.5-flash-lite" \
  --manager-type CodeAgent \
  --tools-list hybrid_automaton_image_analysis validate_hybrid_automaton_specification \
  --managed-agents-list data_analysis_expert \
  --max-iterations 5 \
  --train-num 3 \
  --eval-train-num 1 \
  --feedback-top-k 3 \
  --target-error 0.001 \
  --no-improvement-patience 10
```

### Batch Execution

```bash
# run.sh — parallel dataset execution (configurable MAX_PARALLEL, MODEL, etc.)
bash run.sh

# Or use the Python batch runners
python run_scripts/batchrun.py
```

### Parameter Optimization

```bash
# Nevergrad optimizer (LLM extracts params, Nevergrad optimizes)
python optimize_ha_params.py

# SINDy + Nevergrad pipeline
python optimize_sindy_nevergrad.py
```

### HA Evaluation

```python
from utils.Dainarx_code.HA_evaluation import HAEvaluator

evaluator = HAEvaluator(ha_dict, npz_file_path, dt=0.001, total_time=10.0)
results = evaluator(plot_mode='overlay', save_path='output.png', show_plot=False)
```

### Testing

Tests live in `testfiles/`:
```bash
python testfiles/test_imgTools_ha.py     # Image tool
python testfiles/test_reviewTools_ha.py  # Review tool
python testfiles/test_litellm.py         # API connectivity
```

### Phoenix Tracing (Optional)

```bash
python -m phoenix.server.main serve      # http://localhost:6006
python run_llm_ha_gamma.py --save-traces-dir "phoenix_traces"
```

## Key Data Formats

### Hybrid Automaton JSON

See full spec at `utils/Dainarx_code/automata/json_readme.md`.

```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "u1",
    "mode": [{"id": 1, "eq": "x1[1] = x2[0], x2[1] = -0.1*x2[0] - x1[0]**3 + u1"}],
    "edge": [{"direction": "1 -> 2", "condition": "x1 >= 5", "reset": {"x1": ["", "x1"]}}]
  },
  "config": {"dt": 0.001, "total_time": 10.0, "dim": 2}
}
```

Conventions:
- `x[0]` = value, `x[1]` = first derivative, `x[k]` = k-th derivative
- Left side of ODE must be highest-order derivative
- All variables in `var` must have an ODE

### NPZ Trajectory Data

```python
{
  'state': np.ndarray,         # (num_states, num_steps)
  'input': np.ndarray,         # (num_inputs, num_steps) or (num_steps,)
  'mode': np.ndarray,          # (num_steps,) — mode ID at each step
  'change_points': np.ndarray  # (num_transitions,) — indices of mode switches
}
```

### Data Organization

- Training: `data_all/<benchmark>/<system>/sample_*.npz`
- Ground truth: `data_all/<benchmark>/<system>_g/ground_truth_*.npz`
- Results: `evaluation_results/<benchmark>/<system>/runs/<run_id>/`

## Important Caveats

1. **Input Time Indexing**: `input_array[i]` corresponds to time `(i+1)*dt`, not `i*dt`. The `analyticalInput()` function in `ODE_System.py` handles this.

2. **Mode Numbering**: HA modes are 1-indexed. Ground truth mode IDs match learned IDs after bipartite matching.

3. **Agent Step Limits**: Agents default to `max_steps=80`. Long problems may hit this limit.

4. **Image Resolution**: Tools receive compressed images (~1080px) in initial prompt but can access high-res via `worker_agent.markdown_content_high_res_image`.

5. **Some source files contain Chinese comments** (notably `ha_structured_output.py`).


## Guideline

1. When modifying output formatting or display strings, show the exact expected output format before making changes. Ask for clarification if the desired format isn't explicit.
2. For complex multi-step implementations, use @planning-with-files
3. Use `gemini/gemini-flash-lite-latest` for testing. LiteLLM manages API routing.
4. Always use the CodeAgent class for creating the agents, including the manager agent and the managed agents.
5. Remember, every time you write test code, you should consider all the automata, NOT JUST ONE AUTOMATA. Use the ball as a simple example, and duffing as the complex example.
