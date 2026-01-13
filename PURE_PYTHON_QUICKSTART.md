# Pure Python HA Learning - Quick Start Guide

## TL;DR

```bash
# Run pure Python HA learning (NO JSON!)
python run_pure_python_ha.py \
  --input-data-path data_all/non_linear/duffing \
  --max-iterations 3
```

That's it! The system will:
1. Give LLM a simple Python class template
2. LLM fills in the dynamics
3. Optimizer tunes the parameters
4. System evaluates and provides feedback
5. Repeat until converged

---

## What's Different?

### Old Way (Hybrid)
```
LLM → Python class → to_json() → JSON → HybridAutomata.from_json() → Simulate
```
- Still depends on JSON
- Prompts include 4000+ token JSON Schema
- Two representations to maintain

### New Way (Pure Python)
```
LLM → Python class → Execute → Simulate
```
- **No JSON anywhere**
- Prompts are 54% shorter
- Direct execution (like SR-Scientist)

---

## Quick Examples

### Basic Usage

```bash
# Simplest form (uses defaults)
python run_pure_python_ha.py --input-data-path data_all/non_linear/duffing
```

### With Custom Settings

```bash
# More iterations, more optimization steps
python run_pure_python_ha.py \
  --input-data-path data_all/non_linear/duffing \
  --max-iterations 5 \
  --optimization-iters 200 \
  --target-error 0.005
```

### With Different Model

```bash
# Use a more powerful model
python run_pure_python_ha.py \
  --input-data-path data_all/non_linear/duffing \
  --manager-model gemini/gemini-2.5-pro \
  --max-iterations 3
```

### With Tools

```bash
# Enable vision analysis and validation
python run_pure_python_ha.py \
  --input-data-path data_all/non_linear/duffing \
  --tools-list hybrid_automaton_image_analysis validate_hybrid_automaton_specification \
  --max-iterations 3
```

---

## What Happens

### Iteration 1

**System provides LLM**:
- Trajectory plots
- Simple template with TODOs
- Documentation

**LLM generates**:
```python
class HybridAutomaton:
    def __init__(self):
        self.params = [-0.5, -5.0, 1.0, ...]  # Initial guesses
        self.var = "x1"
        self.input = "u1"
        self.order = 2

    def num_modes(self):
        return 1  # Single mode

    def mode_dynamics(self, mode_id, x, u):
        # Damped harmonic oscillator with input
        return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"

    def guard_condition(self, source, target, x, u):
        return False  # No mode switching

    def reset_map(self, source, target, x, u):
        pass  # No resets
```

**System**:
1. ✓ Executes class
2. ✓ Optimizes params → `[-0.48, -4.95, 0.98]`
3. ✓ Simulates → error = 0.05
4. ✓ Generates feedback

### Iteration 2

**LLM receives**:
- Previous class + optimized params
- Metrics: `max_diff=0.05`
- Feedback: "Try adding nonlinear term"

**LLM adds cubic term**:
```python
def mode_dynamics(self, mode_id, x, u):
    # Added x³ term for Duffing dynamics
    return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + {self.params[3]}*u['u1']"
```

**Result**: Error drops to 0.01 → Done!

---

## Output

After running, you'll find:

```
evaluation_results/
├── iter_1/
│   ├── ha_eval_*.png          # Comparison plot
│   └── results_*.txt           # Metrics + feedback
├── iter_2/
│   └── ...
└── best/
    ├── best_ha_class.py       # Best Python class
    └── best_params.npy        # Best parameters

task_prompts/
├── iter_1/task.md             # What LLM saw
└── iter_2/task.md
```

---

## Command-Line Options

### Required
- `--input-data-path`: Directory with `.npz` trace data files

### Optional (with good defaults)
- `--manager-model`: LLM model (default: `gemini/gemini-flash-lite-latest`)
- `--max-iterations`: Refinement iterations (default: 3)
- `--optimization-iters`: Parameter optimization steps (default: 100)
- `--optimizer-type`: Algorithm (default: `simulated_annealing`)
- `--tools-list`: Tools for agent (default: `[image_analysis, validate]`)
- `--target-error`: Early stop threshold (default: 0.01)
- `--feedback-top-k`: Top results in feedback (default: 3)

---

## Troubleshooting

### "No .npz files found"
- Check that your data directory contains `.npz` files
- Example: `data_all/non_linear/duffing/sample_0.npz`

### "Failed to extract Python class"
- LLM didn't generate proper class format
- Try with a more capable model (`gemini-2.5-pro`)
- Enable validation tool

### "Optimization failed"
- Initial class had syntax errors
- Enable `validate_hybrid_automaton_specification` tool
- Check logs for specific error

### High error after multiple iterations
- System may be genuinely complex
- Try more iterations: `--max-iterations 10`
- Try more optimization: `--optimization-iters 200`
- Enable vision tool to help LLM analyze plots

---

## Tips for Best Results

1. **Start simple**: Use defaults first
2. **Check plots**: Look at `evaluation_results/iter_*/ha_eval_*.png`
3. **Read feedback**: Check `results_*.txt` to understand what's happening
4. **Enable tools**: Vision and validation help a lot
5. **Be patient**: First iteration might not be perfect

---

## Comparison: Pure Python vs Old Approach

| Feature | Pure Python | Hybrid (Phase 4) |
|---------|-------------|------------------|
| **JSON needed?** | No | Yes |
| **Prompt size** | ~3000 tokens | ~6500 tokens |
| **Complexity** | Low | Medium |
| **Debugging** | Easy (pure Python) | Harder (mixed) |
| **Speed** | Fast | Slower |

**Pure Python is simpler, faster, and cleaner!**

---

## Next Steps

After successful run:

1. **Check best result**:
   ```bash
   cat evaluation_results/best/best_ha_class.py
   ```

2. **Load parameters**:
   ```python
   import numpy as np
   params = np.load('evaluation_results/best/best_params.npy')
   print(params)
   ```

3. **Use the class**:
   ```python
   # Execute the best class
   exec(open('evaluation_results/best/best_ha_class.py').read())
   ha = HybridAutomaton()

   # Override with optimized params
   ha.params = params.tolist()

   # Now you can use it!
   ```

---

## FAQ

**Q: Do I need the old `run_llm_ha_beta.py`?**
A: No! `run_pure_python_ha.py` is standalone and self-contained.

**Q: Can I still use JSON format?**
A: Yes, keep using `run_llm_ha_beta.py` without `--use-python-class-format`. But pure Python is recommended.

**Q: What if I have my own NPZ data?**
A: Just point `--input-data-path` to your directory. System auto-detects dimensions.

**Q: How do I know it's working?**
A: Watch for:
- ✓ symbols in console output
- Decreasing error values across iterations
- Comparison plots showing good fit

**Q: Can I stop and resume?**
A: Not yet, but feedback is saved so you can manually continue.

---

## Example Session

```bash
$ python run_pure_python_ha.py --input-data-path data_all/non_linear/duffing --max-iterations 2

================================================================================
PURE PYTHON CLASS-BASED HA LEARNING
================================================================================
Model: gemini/gemini-flash-lite-latest
Data: data_all/non_linear/duffing
Max iterations: 2
Optimization: 100 iters using simulated_annealing
================================================================================

Auto-detected: 1 variable(s), 1 input(s)
Creating agent...
✓ Agent created with 2 tool(s)

################################################################################
ITERATION 1/2
################################################################################

[1/4] Generating task...
✓ Task generated (9613 chars)

[1/4] Running agent...
✓ Agent completed

[1/4] Extracting Python class...
✓ Python class extracted and validated

[1/4] Evaluating...

[1/4] Executing Python class...
✓ Class executed successfully
  - Variables: x1
  - Inputs: u1
  - Modes: 1
  - Order: 2

[2/4] Optimizing parameters (100 iterations)...
✓ Optimization complete
  - Initial error: 5.234567
  - Final error: 0.052341
  - Improvement: 99.00%

[3/4] Simulating with optimized parameters...
✓ Simulation complete

[4/4] Generating feedback...
✓ Feedback generated

Results saved to: evaluation_results/iter_1

✓ Evaluation succeeded
Error: 0.052341

################################################################################
ITERATION 2/2
################################################################################

... (similar) ...

Final error: 0.008234

🛑 Early stop: Target error achieved: 0.008234 < 0.01

================================================================================
EXPERIMENT COMPLETE
================================================================================

✓ Best error: 0.008234 (Iteration 2)
Best class saved to: evaluation_results/best/best_ha_class.py
Best params saved to: evaluation_results/best/best_params.npy

Iteration summary:
  Iter 1: [✓] Error=0.052341
  Iter 2: [✓] Error=0.008234 (BEST)
```

Done! 🎉
