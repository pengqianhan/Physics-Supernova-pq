# HA-Scientist: Hybrid Automaton System Identification Agent

**HA-Scientist** is an agentic workflow designed to autonomously identify Hybrid Automaton (HA) models from time-series trajectory data. It iteratively refines mathematical models to capture the switching dynamics of unknown physical systems.

## Key Features

- **"Scientist-Critic" Loop**: An iterative process where the agent acts as a scientist proposing models and a critic evaluating them against ground truth data.
- **Visual & Numerical Fusion**:
  - **Visual Analysis**: Uses `hybrid_automaton_image_analysis` to detect mode switches and non-smooth behaviors from plots.
  - **Numerical Verification**: Curve fitting and residual analysis.
- **Pure Python Class Generation**: Generates HA classes inheriting from `HybridAutomatonBase` for consistent interface.
- **Parameter Optimization**: Uses simulated annealing or hill climbing to optimize model parameters.
- **Reasoning Log Export**: Saves complete reasoning traces for external monitoring and prompt optimization.

---

## Installation

Requires Python 3.10+.

```bash
python -m pip install -U pip
pip install tenacity smolagents[litellm] loguru python-dotenv
```

### Set up API Keys
Create a `.env` file:
```bash
GEMINI_API_KEY=...    # Google Gemini API Key
```

---

## Usage

The main entry point is `run_pure_python_ha.py`.

### Basic Command
```bash
python run_pure_python_ha.py \
  --input-data-path data_all/ATVA/ball \
  --manager-model gemini/gemini-flash-lite-latest \
  --max-iterations 3 \
  --optimization-iters 50
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-data-path` | `data_all/ATVA/ball` | Directory containing `.npz` trace data files |
| `--manager-model` | `gemini/gemini-flash-lite-latest` | LLM model ID for the agent |
| `--max-iterations` | `3` | Maximum refinement iterations |
| `--optimization-iters` | `50` | Parameter optimization iterations |
| `--optimizer-type` | `simulated_annealing` | Optimizer: `simulated_annealing` or `hill_climbing` |
| `--target-error` | `0.01` | Early stop threshold |
| `--tools-list` | `hybrid_automaton_image_analysis`, `validate_hybrid_automaton_specification` | Tools available to agent |
| `--feedback-top-k` | `3` | Number of top results to include in feedback |
| `--save-reasoning-log` | `reasoning_logs` | Directory for reasoning logs (empty string to disable) |

### Output Structure

```
evaluation_results/
├── iter_1/                    # Per-iteration results
│   ├── trajectory_comparison.png
│   └── metrics.json
├── iter_2/
├── best/                      # Best result
│   ├── best_ha_class.py       # Best HA class code
│   └── best_params.npy        # Optimized parameters

task_prompts/
├── iter_1/task.md             # Generated task prompts
├── iter_2/task.md

reasoning_logs/                # Reasoning traces (NEW)
├── master_reasoning_log.json  # All iterations summary
├── iter_1_reasoning.json      # Iteration 1 detailed JSON
├── iter_1_reasoning.md        # Iteration 1 readable Markdown
├── iter_2_reasoning.json
└── iter_2_reasoning.md
```

---

## Reasoning Log Feature

The reasoning log feature exports complete LLM reasoning traces for external monitoring and prompt optimization.

### Enable/Disable

```bash
# Enable (default)
python run_pure_python_ha.py --save-reasoning-log reasoning_logs

# Custom directory
python run_pure_python_ha.py --save-reasoning-log my_logs

# Disable
python run_pure_python_ha.py --save-reasoning-log ""
```

### File Contents

#### `master_reasoning_log.json`

A JSON array containing all iterations, suitable for monitoring the entire experiment:

```json
[
  {
    "iteration": 1,
    "timestamp": "2025-01-15T10:30:00",
    "input_prompt": "Complete task prompt...",
    "reasoning_steps": [
      {
        "type": "ActionStep",
        "llm_output": "LLM's reasoning and code...",
        "tool_calls": "[...]",
        "observations": "Tool results..."
      }
    ],
    "final_result": "Agent output",
    "extracted_class_code": "class HybridAutomaton...",
    "evaluation_metrics": {"max_diff": 0.05},
    "llm_critique": "Improvement suggestions..."
  }
]
```

#### `iter_N_reasoning.json`

Same content as corresponding entry in `master_reasoning_log.json`, but as a single JSON object.

#### `iter_N_reasoning.md`

Human-readable Markdown version with formatted sections:
- Input Prompt
- Reasoning Steps (LLM outputs, tool calls, observations)
- Extracted Class Code
- Evaluation Metrics
- LLM Critique

### Use Case: External Agent Monitoring

You can use another agent (e.g., Claude Code) to monitor and optimize prompts:

```bash
# Watch the master log in real-time
tail -f reasoning_logs/master_reasoning_log.json

# Query error trends
cat reasoning_logs/master_reasoning_log.json | jq '.[].evaluation_metrics.max_diff'

# Read specific iteration
cat reasoning_logs/iter_1_reasoning.md
```

---

## Data Format

### NPZ Trajectory Data

```python
{
  'state': np.ndarray,         # Shape: (num_states, num_steps)
  'input': np.ndarray,         # Shape: (num_inputs, num_steps)
  'mode': np.ndarray,          # Shape: (num_steps,) - mode ID at each step
  'change_points': np.ndarray  # Shape: (num_transitions,) - mode switch indices
}
```

---

## Code Reference: Physics Supernova

This project builds upon the architecture of **Physics Supernova**, an agent system capable of solving elite-level Physics problems.

### Citation

If you find this work useful, please cite the Physics Supernova paper:

```bibtex
@misc{qiu2025physicssupernovaaiagent,
      title={Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025},
      author={Jiahao Qiu and Jingzhe Shi and Xinzhe Juan and Zelin Zhao and Jiayi Geng and Shilong Liu and Hongru Wang and Sanfeng Wu and Mengdi Wang},
      year={2025},
      eprint={2509.01659},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.01659},
}
```

### Acknowledgments
Built on [`smolagents`](https://github.com/huggingface/smolagents).
