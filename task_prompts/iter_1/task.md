# HYBRID AUTOMATON IDENTIFICATION TASK

## Your Role
You are a control systems engineer identifying a Hybrid Automaton from trajectory data.

## System Information
- **State variables**: x1 (1 variable)
- **Inputs**: u1 (1 input)
- **Iteration**: 1

## Objective
Generate a complete Python class implementing a Hybrid Automaton that accurately reproduces the observed dynamics.

# Hybrid Automaton Python Class Specification

## Core Concept
A **Hybrid Automaton** models systems with:
- **Discrete modes**: Different operating regimes (e.g., "free fall" vs "bouncing")
- **Continuous dynamics**: ODEs governing each mode
- **Mode switching**: Guard conditions triggering transitions
- **State resets**: Optional jumps when switching modes

## Required Class Structure

```python
class HybridAutomaton:
    def __init__(self):
        # Tunable numerical parameters (will be optimized)
        self.params = [p0, p1, p2, ...]  # Max 10 parameters

        # System metadata (FIXED - inferred from data)
        self.var = "x1, x2, ..."     # State variable names
        self.input = "u1, u2, ..."   # Input names (or "" if none)
        self.dt = 0.001              # Time step
        self.total_time = 10.0       # Simulation duration
        self.order = 1 or 2          # ODE order

    def num_modes(self) -> int:
        """Return number of discrete modes."""
        return N  # You determine N from data

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        """
        Return ODE equation string for mode_id.

        Args:
            mode_id: Mode number (1, 2, 3, ...)
            x: State dict {'x1': [x1_value, x1_dot, ...], ...}
            u: Input dict {'u1': value, ...}

        Returns:
            Equation string (e.g., "x1[2] = params[0]*x1[1] + params[1]*x1[0] + u['u1']")
        """
        if mode_id == 1:
            return "..."  # Your equation here
        elif mode_id == 2:
            return "..."
        # ... more modes

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        """Return True if transition should occur."""
        if source_mode == 1 and target_mode == 2:
            return x['x1'][0] >= threshold  # Your condition
        # ... more guards
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        """Modify x in-place if state reset is needed."""
        # Example: x['x1'][0] = -0.9 * x['x1'][0]
        pass  # Leave empty if no resets
```

## Key Rules

1. **Variable Count is FIXED**: `self.var` and `self.input` are pre-determined from data
   - DO NOT change the number of variables
   - For single-variable systems, use higher-order ODEs (e.g., `x1[2] = ...` for 2nd order)
   - DO NOT split into multiple 1st-order variables

2. **ODE Notation**:
   - `x[0]` = variable value
   - `x[1]` = first derivative (dx/dt)
   - `x[2]` = second derivative (d²x/dt²)
   - Left side = highest derivative (e.g., `x1[2] = ...` for order=2)

3. **Parameters**:
   - Store ALL numerical values in `self.params` list
   - Use `self.params[i]` in equations (optimizer will tune these)
   - Max 10 parameters (params[0] through params[9])

4. **Mode IDs**: Start from 1 (not 0)

5. **Equation Strings**:
   - Must be valid Python expressions
   - Reference params via `{self.params[i]}` in f-strings
   - Access inputs as `u['input_name']`
   - Example: `f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + u['u1']"`

## Common Patterns

### Single-Mode Oscillator (Damped + Forced)
```python
def __init__(self):
    self.params = [-0.5, -5.0, 1.0, 0, 0, 0, 0, 0, 0, 0]
    self.var, self.input, self.order = "x1", "u1", 2

def num_modes(self): return 1

def mode_dynamics(self, mode_id, x, u):
    # x'' = damping*x' + stiffness*x + gain*u
    return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"
```

### Two-Mode Switching System
```python
def __init__(self):
    self.params = [2.0, -3.0, 1.5, 5.0, 0, 0, 0, 0, 0, 0]
    # [k1, k2, threshold, ...]
    self.var, self.input, self.order = "x1", "", 1

def num_modes(self): return 2

def mode_dynamics(self, mode_id, x, u):
    if mode_id == 1:
        return f"x1[1] = {self.params[0]}"  # Constant rate mode 1
    elif mode_id == 2:
        return f"x1[1] = {self.params[1]}"  # Constant rate mode 2

def guard_condition(self, source, target, x, u):
    if source == 1 and target == 2:
        return x['x1'][0] >= self.params[2]  # Switch up
    elif source == 2 and target == 1:
        return x['x1'][0] < self.params[3]  # Switch down
    return False
```

### Nonlinear System (e.g., Duffing)
```python
def mode_dynamics(self, mode_id, x, u):
    # x'' + damping*x' + k1*x + k3*x³ = u
    return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + u['u1']"
```

## Your Task

You will be given:
1. **Trajectory plots** showing state evolution over time
2. **Raw data** (optional - for detailed analysis)
3. **Initial template** with correct `var`, `input`, `order` pre-filled

Your job:
1. **Analyze** the trajectory patterns (oscillations? jumps? mode switches?)
2. **Infer** the number of modes, dynamics equations, and switching logic
3. **Generate** a complete Python class implementing the Hybrid Automaton
4. **Iterate** based on evaluation feedback to improve accuracy

## Evaluation Metrics

Your specification will be scored on:
- **max_diff**: Maximum state error vs ground truth (lower is better, < 0.01 is good)
- **mean_diff**: Average state error (< 0.005 is good)
- **tc (change-point error)**: Mode switch timing accuracy (< 0.01s is good)

After each iteration, you'll receive:
- Metrics from evaluation
- Comparison plot (your simulation vs ground truth)
- Feedback on what to improve

Use this to refine your class in the next iteration!



## Available Resources
**Tools**: You have access to: hybrid_automaton_image_analysis, validate_hybrid_automaton_specification
- Use `hybrid_automaton_image_analysis` to get vision model insights on trajectory patterns
- Use `validate_hybrid_automaton_specification` to check your class syntax before submission

**Code Execution**: You can execute Python code to analyze data or test ideas

## Data Sources
- **Trajectory plots**: 3 visualization(s) available
- **Raw data files**: ['<npz_0>', '<npz_1>', '<npz_2>']

## Starting Template (v0)

Here's a minimal template with the correct structure. Your job is to fill in the correct:
- Number of modes (`num_modes()`)
- Dynamics equations (`mode_dynamics()`)
- Guard conditions (`guard_condition()`)
- Reset logic (`reset_map()` if needed)

```python
class HybridAutomaton:
    """
    Hybrid Automaton specification.

    TODO: Infer the correct dynamics, modes, and switching conditions from data.
    """

    def __init__(self):
        # Numerical parameters (tune these to fit the data)
        self.params = [
            0.0,  # params[0]: placeholder
            0.0,  # params[1]: placeholder
            0.0,  # params[2]: placeholder
            0.0,  # params[3]: placeholder
            0.0,  # params[4]: placeholder
            0.0, 0.0, 0.0, 0.0, 0.0  # params[5-9]: unused
        ]

        # System structure (FIXED - do not change these)
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2

    def num_modes(self) -> int:
        """Return number of discrete modes."""
        # TODO: Determine correct number of modes from data
        return 1

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        """
        Return ODE equation string for the given mode.

        Args:
            mode_id: Mode identifier (1-indexed)
            x: State dict {var_name: [x[0], x[1], ...]}
            u: Input dict {input_name: value}

        Returns:
            ODE equation string (e.g., "x1[2] = params[0]*x1[1] + params[1]*x1[0] + u1")
        """
        if mode_id == 1:
            # TODO: Replace with correct dynamics
            # For order=2 system, left side should be highest derivative
            # Example 2nd-order dynamics
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0]"
        else:
            raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        """
        Evaluate guard condition for mode transition.

        Returns True if transition should occur.
        """
        # TODO: Add switching conditions if multiple modes exist
        # Example: if source_mode == 1 and target_mode == 2:
        #              return x['x1'][0] >= self.params[3]
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        """
        Apply state reset on transition (modifies x in-place).

        Leave empty if no resets needed.
        """
        # TODO: Add reset logic if needed
        # Example: x['x1'][0] = -0.9 * x['x1'][0]
        pass

```

**Replace the TODO placeholders with your inferred dynamics!**


## Your Output

Generate a **complete, executable Python class** that:
1. Accurately models the observed dynamics
2. Uses `self.params` for all tunable numerical values
3. Implements all required methods (`__init__`, `num_modes`, `mode_dynamics`, `guard_condition`, `reset_map`)

**Important**:
- Keep `self.var = "x1"` FIXED
- Keep `self.input = "u1"` FIXED
- Provide reasonable initial guesses for `self.params` (they will be auto-optimized)

Output only the Python class code, no explanations before or after.
