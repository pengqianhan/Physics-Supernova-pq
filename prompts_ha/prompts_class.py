"""
Python class-based Hybrid Automaton format documentation for LLM guidance.

This module provides comprehensive documentation and examples to guide the LLM
in generating valid Python class-based HA specifications with optimizable parameters.
"""

# ============================================================================
# Main Documentation
# ============================================================================

HA_CLASS_DOCUMENTATION = """
## Hybrid Automaton Python Class Format

You are tasked with identifying a Hybrid Automaton (HA) model from trajectory data.
Instead of JSON format, you will generate an executable Python class with optimizable parameters.

### 🎯 GOAL: PARSIMONIOUS SYSTEM IDENTIFICATION
Your objective is to identify the **simplest possible** Hybrid Automaton that explains the data.
- **Penalty**: You will be penalized for unnecessary modes or complex nonlinear terms.
- **Strategy**: Start with a single mode with linear dynamics. Only add complexity (modes/nonlinearity) if error remains high.

### ⚠️ CRITICAL: Use Global Params Array

ALL numerical parameters MUST be stored in `self.params` list (maximum 10 elements):
- ✅ **CORRECT**: `{self.params[0]} * x1[1]`
- ❌ **WRONG**: Hardcoded `0.5 * x1[1]`

**Why?** After you generate the class structure, a gradient-free optimizer will automatically
tune the parameter values to minimize error. Hardcoded values cannot be optimized!

### Required Class Structure

```python
class HybridAutomaton:
    \"\"\"
    Hybrid Automaton specification with optimizable parameters.

    Attributes:
        params: Global array of numerical parameters (max 10)
        var: State variable names (comma-separated)
        input: Input variable names (comma-separated)
        dt, total_time, order: Simulation configuration
    \"\"\"

    def __init__(self):
        # REQUIRED: Initialize params list with your best guesses
        self.params = [
            -0.5,  # params[0]: damping coefficient
            -5.0,  # params[1]: spring constant
            1.0,   # params[2]: input gain
            # ... up to params[9]
        ]

        # REQUIRED: System structure
        self.var = "x1"           # State variables (comma-separated)
        self.input = "u1"         # Input variables (comma-separated or empty "")
        self.dt = 0.001           # Time step
        self.total_time = 10.0    # Simulation duration
        self.order = 2            # ODE order (1 or 2)
        self.need_reset = False   # Whether state resets occur
        self.non_linear_items = ""  # Nonlinear terms (if any)

    def num_modes(self) -> int:
        \"\"\"Return number of discrete modes.\"\"\"
        return 1  # Change based on your model

    def mode_dynamics(self, mode_id: int, x: dict, u: dict) -> str:
        \"\"\"
        Return ODE equation string for the given mode.

        Args:
            mode_id: Mode identifier (1-indexed)
            x: State dict {var_name: [x[0], x[1], ...]}
            u: Input dict {input_name: value}

        Returns:
            ODE equation string using x[k] notation:
            - x1[0]: current value
            - x1[1]: first derivative
            - x1[2]: second derivative
        \"\"\"
        # Example for mode 1
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode: int, target_mode: int, x: dict, u: dict) -> bool:
        \"\"\"
        Evaluate guard condition for mode transition.

        Returns True if transition should occur.
        \"\"\"
        # Example: No transitions for single-mode system
        return False

    def reset_map(self, source_mode: int, target_mode: int, x: dict, u: dict):
        \"\"\"
        Apply state reset on transition (modifies x in-place).

        Leave empty (pass) if no resets occur.
        \"\"\"
        pass

    def to_json(self) -> dict:
        \"\"\"
        Convert to JSON format for compatibility with the simulator.

        IMPORTANT: This method is auto-generated based on your mode_dynamics,
        guard_condition, and reset_map methods. You MUST implement it.
        \"\"\"
        # Build mode equations by calling mode_dynamics
        mode_1_eq = self.mode_dynamics(1, {}, {}).replace("u['u1']", "u1")

        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [
                    {"id": 1, "eq": mode_1_eq}
                ],
                "edge": []  # Empty for single-mode system
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
```

### 📐 ODE Format Rules

**Higher-Order ODE Format** (NOT state-space form):
- For a second-order system with 1 state variable: `x1[2] = ...`
- For a first-order system with 2 state variables: `x1[1] = ..., x2[1] = ...`

**Variable Notation**:
- `x1[0]`: Current value of x1
- `x1[1]`: First derivative dx1/dt
- `x1[2]`: Second derivative d²x1/dt²

**Common Mistake to Avoid**:
❌ WRONG: Converting 2nd-order ODE to state-space:
```python
self.var = "x1, x2"  # WRONG!
# x1[1] = x2[0], x2[1] = -0.5*x2[0] - 5*x1[0]
```

✅ CORRECT: Use higher-order form directly:
```python
self.var = "x1"  # CORRECT!
# x1[2] = -0.5*x1[1] - 5*x1[0]
```

### 🔧 Params Array Usage

**Initialization**:
- Provide your best guesses for parameter values based on the trajectory data
- Use physics intuition: damping ~0.1-1.0, spring constant ~1-10, etc.
- The optimizer will refine these values automatically

**Indexing**:
- params[0], params[1], ..., params[9] (max 10 parameters)
- Add comments explaining what each parameter represents
- Unused parameters should be set to 0.0

**In Equations**:
- Always use f-string formatting: `f"{self.params[0]}*x1[1]"`
- Do NOT hardcode: `f"0.5*x1[1]"` is WRONG
- Combine params with operations: `f"{self.params[0]*2}*x1[0]"` is OK

### 🔄 Guard Conditions and Transitions

**For Single-Mode Systems**:
```python
def guard_condition(self, source_mode, target_mode, x, u):
    return False  # No transitions

def reset_map(self, source_mode, target_mode, x, u):
    pass  # No resets
```

**For Multi-Mode Systems**:
```python
def guard_condition(self, source_mode, target_mode, x, u):
    if source_mode == 1 and target_mode == 2:
        return x['x1'][0] >= self.params[3]  # Threshold switching
    elif source_mode == 2 and target_mode == 1:
        return x['x1'][0] < self.params[3]
    return False
```

**Guard Syntax**:
- Use `x['var_name'][0]` for current value
- Use `x['var_name'][1]` for first derivative
- Comparisons: `>=`, `<=`, `>`, `<`, `==`
- Logic: `and`, `or`, `not`

### 🔄 to_json() Implementation Guide

The `to_json()` method converts your class to the simulator's JSON format.

**Key Steps**:
1. Call `mode_dynamics()` for each mode to get equation strings
2. Replace `u['u1']` with `u1` (simulator format)
3. Build guard conditions from `guard_condition()` method
4. Build reset maps from `reset_map()` method

**Example for 2-mode system**:
```python
def to_json(self):
    # Get equations for each mode
    mode_1_eq = self.mode_dynamics(1, {}, {}).replace("u['u1']", "u1")
    mode_2_eq = self.mode_dynamics(2, {}, {}).replace("u['u1']", "u1")

    return {
        "automaton": {
            "var": self.var,
            "input": self.input,
            "mode": [
                {"id": 1, "eq": mode_1_eq},
                {"id": 2, "eq": mode_2_eq}
            ],
            "edge": [
                {"direction": "1 -> 2", "condition": f"x1 >= {self.params[3]}"},
                {"direction": "2 -> 1", "condition": f"x1 < {self.params[3]}"}
            ]
        },
        "config": {
            "dt": self.dt,
            "total_time": self.total_time,
            "order": self.order,
            "need_reset": self.need_reset,
            "non_linear_items": self.non_linear_items
        }
    }
```

---

## 📚 Complete Examples

### Example 1: Single-Mode Damped Harmonic Oscillator (2nd-order)

**System**: `d²x/dt² + c*dx/dt + k*x = u`

```python
class HybridAutomaton:
    \"\"\"Damped harmonic oscillator with external input.\"\"\"

    def __init__(self):
        # Optimizable parameters
        self.params = [
            -0.1,  # params[0]: damping coefficient c
            -5.0,  # params[1]: spring constant k
            1.0,   # params[2]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # params[3-9]: unused
        ]

        # System structure
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u1"
        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [{"id": 1, "eq": mode_eq}],
                "edge": []
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
```

---

### Example 2: Two-Mode System with Threshold Switching

**System**: Oscillator with different damping in two modes
- Mode 1 (x < threshold): Light damping
- Mode 2 (x >= threshold): Heavy damping

```python
class HybridAutomaton:
    \"\"\"Two-mode oscillator with threshold-based mode switching.\"\"\"

    def __init__(self):
        # Optimizable parameters
        self.params = [
            -0.1,  # params[0]: damping mode 1
            -5.0,  # params[1]: spring constant (shared)
            -0.5,  # params[2]: damping mode 2
            0.0,   # params[3]: switching threshold
            1.0,   # params[4]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0  # params[5-9]: unused
        ]

        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 2

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            # Light damping
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u['u1']"
        elif mode_id == 2:
            # Heavy damping
            return f"x1[2] = {self.params[2]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        if source_mode == 1 and target_mode == 2:
            return x['x1'][0] >= self.params[3]
        elif source_mode == 2 and target_mode == 1:
            return x['x1'][0] < self.params[3]
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass  # No state resets

    def to_json(self):
        mode_1_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u1"
        mode_2_eq = f"x1[2] = {self.params[2]}*x1[1] + {self.params[1]}*x1[0] + {self.params[4]}*u1"

        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [
                    {"id": 1, "eq": mode_1_eq},
                    {"id": 2, "eq": mode_2_eq}
                ],
                "edge": [
                    {"direction": "1 -> 2", "condition": f"x1 >= {self.params[3]}"},
                    {"direction": "2 -> 1", "condition": f"x1 < {self.params[3]}"}
                ]
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
```

---

### Example 3: Nonlinear System (Duffing Oscillator)

**System**: `d²x/dt² + c*dx/dt + k1*x + k3*x³ = u`

```python
class HybridAutomaton:
    \"\"\"Duffing oscillator with cubic nonlinearity.\"\"\"

    def __init__(self):
        # Optimizable parameters
        self.params = [
            -0.1,  # params[0]: damping c
            -1.0,  # params[1]: linear stiffness k1
            -1.0,  # params[2]: cubic stiffness k3
            1.0,   # params[3]: input gain
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # params[4-9]: unused
        ]

        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = "x1**3"  # Document nonlinear terms

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + {self.params[3]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + {self.params[3]}*u1"

        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [{"id": 1, "eq": mode_eq}],
                "edge": []
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
```

---

## 🚀 Optimization Workflow

After you generate the Python class:
1. **Validation**: Syntax and structure are checked
2. **Optimization**: Gradient-free optimizer tunes params[] values (typically 100 iterations)
3. **Evaluation**: Simulated trajectory compared to ground truth → metrics computed
4. **Feedback**: Optimized params + error metrics returned to you

**Your job**: Design the structure (number of modes, transitions, nonlinear terms)
**Optimizer's job**: Fine-tune numerical parameter values

---

## ⚠️ Common Mistakes to Avoid

1. ❌ Hardcoding numerical values instead of using params:
   ```python
   # WRONG:
   return "x1[2] = 0.5*x1[1] + 2.0*x1[0]"

   # CORRECT:
   return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0]"
   ```

2. ❌ Converting to state-space when single variable:
   ```python
   # WRONG (for 2nd-order system):
   self.var = "x1, x2"

   # CORRECT:
   self.var = "x1"  # Use 2nd-order ODE directly
   ```

3. ❌ Missing to_json() method or incorrect implementation

4. ❌ Params as tuple instead of list:
   ```python
   # WRONG:
   self.params = (1.0, 2.0)

   # CORRECT:
   self.params = [1.0, 2.0]
   ```

5. ❌ Using x[0] notation in guard conditions in to_json():
   ```python
   # WRONG in JSON:
   "condition": "x1[0] >= 0"

   # CORRECT in JSON:
   "condition": f"x1 >= {self.params[3]}"
   ```

---

## 📝 Final Checklist

Before submitting your class, ensure:
- ✅ All numerical values are in self.params[]
- ✅ params is a list (not tuple)
- ✅ All 6 required methods are implemented
- ✅ self.var, self.input, self.dt, self.total_time, self.order are set
- ✅ ODE format uses x[k] notation (x[0]=value, x[1]=derivative)
- ✅ to_json() correctly converts to simulator format
- ✅ Guard conditions use x['var'][k] in Python, bare 'var' in JSON

Good luck! Start simple (single mode, linear dynamics) and add complexity only if needed.
"""


# ============================================================================
# Helper function for prompt generation
# ============================================================================

def get_ha_class_documentation() -> str:
    """
    Get the complete Python class format documentation for HA specifications.

    Returns:
        Formatted documentation string
    """
    return HA_CLASS_DOCUMENTATION


if __name__ == "__main__":
    # Print documentation for review
    print(get_ha_class_documentation())
    print("\n" + "=" * 80)
    print(f"Documentation length: {len(HA_CLASS_DOCUMENTATION)} characters")
    print(f"Documentation lines: {len(HA_CLASS_DOCUMENTATION.split(chr(10)))} lines")
