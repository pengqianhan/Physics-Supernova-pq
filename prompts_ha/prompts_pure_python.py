"""
Simplified prompts for pure Python class-based Hybrid Automaton generation.

No JSON Schema, no conversion - direct Python class generation with inheritance.
Inspired by SR-Scientist's approach.
"""

# Structured output format specification
STRUCTURED_OUTPUT_FORMAT = """
## Output Format

### Analysis Process
[~100 words: observed patterns, mode count, dynamics rationale]

### Python Class
```python
from utils.ha_base_class import HybridAutomatonBase

class HybridAutomaton(HybridAutomatonBase):
    ...
```
"""

# Concise documentation focused on Python class structure with inheritance
HA_PYTHON_CLASS_DOCS = """# Hybrid Automaton Specification

## Class Structure

```python
from utils.ha_base_class import HybridAutomatonBase

class HybridAutomaton(HybridAutomatonBase):
    def __init__(self):
        self.params = [p0, p1, ...]  # Tunable values (max 10)
        self.var = "x1"              # FIXED from data
        self.input = "u1"            # FIXED from data
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2               # 1 or 2

    def num_modes(self) -> int:
        return N  # Infer from data

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        # Return ODE string. x['x1'][0]=value, x['x1'][1]=derivative
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + u['u1']"

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        # Return True when transition should occur
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        # Modify x in-place if needed (e.g., x['x1'][1] = -0.8*x['x1'][1])
        pass
```

## Key Rules
- **Inherit**: `class HybridAutomaton(HybridAutomatonBase)`
- **ODE notation**: `x[0]`=value, `x[1]`=dx/dt, `x[2]`=d²x/dt². Left side = highest derivative
- **Params**: Use `self.params[i]` for all tunable numbers (optimizer will tune)
- **Mode IDs**: Start from 1 (not 0)
- **Variables**: Keep `self.var` and `self.input` FIXED as provided

## Metrics (lower is better)
- `max_diff < 0.01`: good fit
- `mean_diff < 0.005`: accurate overall
- `tc < 0.01s`: mode switch timing correct
"""


def get_python_class_task_prefix(num_variables: int, num_inputs: int, iteration: int = 1) -> str:
    """Generate task prefix for pure Python class generation."""

    var_names = ", ".join([f"x{i+1}" for i in range(num_variables)])
    input_names = ", ".join([f"u{i+1}" for i in range(num_inputs)]) if num_inputs > 0 else "(none)"

    prefix = f"""# Hybrid Automaton Identification - Iteration {iteration}

**System**: {num_variables} variable(s) [{var_names}], {num_inputs} input(s) [{input_names}]

**Task**: Generate a Python class that reproduces the observed dynamics.

{HA_PYTHON_CLASS_DOCS}

"""
    return prefix


def get_feedback_section(feedback: str) -> str:
    """Format feedback from previous iteration."""
    if not feedback:
        return ""

    return f"""
## FEEDBACK FROM PREVIOUS ITERATION

{feedback}

Use this feedback to improve your specification in this iteration!
"""


# Quick test
if __name__ == "__main__":
    print(get_python_class_task_prefix(num_variables=1, num_inputs=1, iteration=1))
    print("\n" + "="*80 + "\n")
    print(get_feedback_section("Previous attempt had max_diff=0.05. Mode dynamics may be incorrect."))
