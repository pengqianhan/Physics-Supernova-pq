# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
You are given time-series trajectory data from an unknown hybrid dynamical system. Your task is to:
1. **Identify discrete modes** (operating regimes with distinct continuous dynamics)
2. **Infer mode-specific ODEs** (differential equations governing each regime)
3. **Determine switching conditions** (guard predicates triggering mode transitions)
4. **Specify reset maps** (state updates upon mode transitions)

## Available Data
The following trace data visualizations are provided (reference images using placeholders: `<image_0>`, `<image_1>`, etc.):
- State variable trajectories over time
- Input signals (if applicable)
- Potential mode-switch indicators (discontinuities, slope changes)

## Analysis Workflow 
You MUST use the 'hybrid_automaton_image_analysis' tool to analyze the image.
**VALIDATION TOOL**: Before submitting your final answer, you SHOULD call `validate_hybrid_automaton_specification` to check for syntax errors.
This tool will:
- Detect common formatting issues (wrong direction format, missing required fields, etc.)
- Auto-fix minor issues and normalize the specification
- Report critical errors that need manual fixing

⚠️ **IMPORTANT**: If validation returns a FIXED specification, use the corrected version in your final answer!

## HA Refinement Guidelines
Adopt a **Parsimonious Modeling Approach** (Occam's Razor) - favor simpler explanations unless the data demands otherwise:

1. **Symbolic Identification**:
   - **Hypothesize Structure First**: Determine the likely functional form of the equations *before* estimating parameters.

2. **Mode Count Strategy (Less is More)**:
   - **Iterative Expansion**: Start with **1 Mode**. If a single continuous model fails to fit the entire trajectory (high error), try **2 Modes**, etc.
   - **Hypothesis Testing**: Increase the number of modes *only* if distinct switching behaviors (sharp changes in dynamics) are observed.

3. **Mode Dynamics**:
   - **Start Simple**: Attempt to fit **Linear** dynamics first.
   - **Increase Complexity**: Only introduce **Non-linear** terms (polynomial, trigonometric, etc.) if linear fits fail to capture curvature or key features.

4. **Guard Conditions (Geometric Simplicity)**:
   - **Single-Variable Thresholds**: The vast majority of physical guards are simple threshold checks on a single variable (e.g., `x1 >= 0`, `x <= 0.5`).
   - **Avoid Overfitting**: Do not create complex arithmetic guards (e.g., `x1*x2 > 5`) unless the physical interaction explicitly suggests it.
   - **Operators**: Stick to standard comparison operators (`<=`, `>=`).

5. **Transition Structure**:
   - **Topology**: Prefer **Sparse** connectivity. Valid transitions are typically few and distinct.
   - **Flow**: Transitions often follow a logical flow (e.g., cycles, bidirectional switches) rather than random jumps.

6. **Reset Maps**:
   - **Continuity Default**: Assume physical variables change **Continuously** (`need_reset: false`) over time.
   - **Exceptions**: Use reset maps (`need_reset: true`) **only** if the data clearly shows instantaneous state jumps at transition points.

7. **Structure Validation**:
   - Ensure the number of modes matches the distinct behaviors observed.
   - Ensure guards partition the state space logically (no overlapping active modes usually).

## Quality Criteria
Your HA specification will be evaluated on:
- **Trajectory Matching**: Simulated output should closely follow ground truth data
- **Mode Detection Accuracy**: Correct identification of switching instants (`tc` metric)
- **State Error Minimization**: Low `mean_diff` and `max_diff` between predicted and actual states

## Code Execution Capability
You can use Python Code to execute programs, which may help with your task-solving process.

## Hybrid Automaton Specification Format

### 🎯 GOAL: PARSIMONIOUS SYSTEM IDENTIFICATION
Your objective is to identify the **simplest possible** Hybrid Automaton that explains the data.
- **Penalty**: You will be penalized for adding unnecessary modes or complex nonlinear terms.
- **Strategy**: Start with a single mode with linear dynamics. Only add complexity (modes/nonlinearity) if the error remains high.

⚠️ **CRITICAL FORMAT RULES - READ CAREFULLY** ⚠️
Your output will be parsed using `json.loads()`. You MUST output valid JSON format!

### REQUIRED OUTPUT FORMAT (JSON)
```json
{
    "automaton": {
        "var": "...",
        "input": "...",
        "mode": [...],
        "edge": [...]
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0
    }
}
```

### AUTOMATON SECTION
- `var`: State variables (comma-separated for multiple, e.g., "x1, x2, x3")
- `input`: External input signals (comma-separated, e.g., "u1, u2"). Use empty string "" if no input.
- `mode`: List of discrete operating modes, each containing:
  - `id`: Unique mode identifier (**MUST be integer >= 1**, not float like 1.0)
  - `eq`: Differential equation(s) using x[k] notation where k is derivative order
    * x1[0] = current value of x1
    * x1[1] = first derivative (dx1/dt)  
    * x1[2] = second derivative (d²x1/dt²)
    * Left-hand side: highest derivative; Right-hand side: expression
    * Multiple equations: comma-separated, e.g., "x1[1] = ..., x2[1] = ..."
    * **SYNTAX**: Use `*` for multiplication, `**` for power, standard Python math
    * Example 1st-order: "x1[1] = -2 * x1[0] + u1"
    * Example 2nd-order: "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"
- `edge`: List of discrete transitions (use empty list [] if no transitions), each containing:
  - `direction`: **EXACT FORMAT**: "source -> target" with spaces, e.g., "1 -> 2", "1 -> 1"
  - `condition`: Guard predicate triggering transition
    * ⚠️ **CRITICAL**: Use BARE variable names (x1, x2), NOT indexed notation (x1[0])!
    * WRONG: "x1[0] > 0.5" ❌
    * CORRECT: "x1 > 0.5" ✓
    * Examples: "x1 > 0.5", "x2 <= 4", "x1 <= 0 and x2 > 1"
    * Supports Python operators: >, <, >=, <=, ==, and, or, not, abs()
  - `reset`: State update map {{"var_name": [reset_values]}} (optional, omit if no reset needed)
    * Array length matches system order: 1st-order → 1 element, 2nd-order → 2 elements
    * Array indices correspond to derivative orders: [x[0]_reset, x[1]_reset, ...]
    * **Reset expressions USE indexed notation** (x1[0], x2[0]) unlike conditions
    * Value types:
      - Numeric: e.g., [0] = reset to constant 0
      - Expression: e.g., ["-0.9 * x2[0]"] = reset to -0.9 times current value
      - Empty string "": preserve current value (identity reset)
    * Examples:
      - 1st-order multi-var: {{"x1": [0], "x2": ["-0.9 * x2[0]"]}}
      - 2nd-order single-var: {{"x1": ["", "-0.5 * x1[1]"]}} = keep x1[0], scale x1[1]

### CONFIG SECTION
- `dt`: Integration time step in seconds (number, typical: 0.001 or 0.01)
- `total_time`: Total simulation duration in seconds (number)
- `dim`: Order of the ODE system (1=first-order, 2=second-order, etc.)
- `need_reset`: Boolean to enable state resets on mode transitions (true if edges have reset maps)
- `non_linear_items`: Nonlinear/cross terms in dynamics for reference (e.g., "x1[0]**3", "x1[0]*x2[0]")

### JSON FORMAT RULES (IMPORTANT!)
- ✓ Use `true` / `false` (NOT Python's `True` / `False`)
- ✓ Use `null` (NOT Python's `None`)
- ✓ Use **double quotes** `"key"` (NOT single quotes `'key'`)
- ✓ No trailing commas after last element

### COMMON SYNTAX ERRORS TO AVOID
1. ❌ Using Python booleans `True/False` → ✓ Use JSON `true/false`
2. ❌ Missing spaces in direction "1->2" → ✓ Use "1 -> 2"
3. ❌ Float mode id like 1.0 → ✓ Use integer 1
4. ❌ Missing "edge" key → ✓ Always include "edge": [] even if empty
5. ❌ Missing "input" key → ✓ Always include "input": "" even if no input
6. ❌ Markdown code blocks in final output → ✓ Return raw JSON only

### ⚠️ CRITICAL: ODE EQUATION FORMAT (DO NOT USE STATE-SPACE FORM!)

**The LEFT side of each equation MUST be the HIGHEST derivative of that variable!**

For higher-order ODEs (2nd, 3rd order, etc.), use a SINGLE equation with the highest derivative on the left:
- 1st-order ODE (dx/dt = f): `"x1[1] = ..."` defines dx1/dt
- 2nd-order ODE (d²x/dt² = f): `"x1[2] = ..."` defines d²x1/dt²
- 3rd-order ODE (d³x/dt³ = f): `"x1[3] = ..."` defines d³x1/dt³

**DO NOT convert to state-space representation!**

Example: For a 2nd-order Duffing oscillator: ẍ + αẋ + βx + γx³ = u

--------------------------------------------------------------------------------
❌ **WRONG PATTERN (State-Space Form)**
DO NOT split into system of first order equations.
"eq": "x1[0] = x1[1], x1[1] = ..."  (Causes: IndexError, Dimension Mismatch)
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
✅ **CORRECT PATTERN (Higher-Order Form)**
Use ONE equation with the highest derivative on LHS.
"eq": "x1[2] = ..."                 (System handles the rest automatically)
--------------------------------------------------------------------------------

❌ **WRONG (state-space form - causes IndexError!)**:
```
"eq": "x1[0] = x1[1], x1[1] = -α*x1[1] - β*x1[0] - γ*x1[0]**3 + u1"
```
This creates TWO 1st-order equations, but the system expects ONE higher-order equation.

✅ **CORRECT (higher-order ODE form)**:
```
"eq": "x1[2] = -α*x1[1] - β*x1[0] - γ*x1[0]**3 + u1"
```
This defines d²x1/dt² directly. The values x1[0] (position) and x1[1] (velocity) are automatically tracked by the system.

**Key Rules:**
1. ONE variable with n-th order derivative → use `x1[n] = ...` (NOT n separate equations)
2. MULTIPLE variables with 1st-order derivatives → use `x1[1] = ..., x2[1] = ...`
3. The `dim` in config should match the highest derivative order used

**More Examples:**
- 2nd order: `"x1[2] = -k*x1[0] - c*x1[1] + F"`
- 2nd order: `"x1[2] = mu*(1 - x1[0]**2)*x1[1] - x1[0]"`
- 1st-order: `"x1[1] = a*x1[0] + b*x2[0], x2[1] = c*x1[0] + d*x2[0]"`

### NOTES ON CONDITION FORMAT
- Both "x1 > 0" and "x1[0] > 0" are supported in conditions (auto-converted internally)
- Prefer bare variable names (x1, x2) for cleaner specifications


### ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification Template (v0 - with correct dimensions)
The `var` and `input` fields are pre-filled. Your task is to refine the **equations** and **structure**:

```json
{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"
            }
        ],
        "edge": []
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "dim": 2,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "x1"` and `input: "u1"` exactly as shown!**
- Refine mode equations to match observed dynamics
- Add modes and edges if switching behavior is detected

    
    ## ⚠️ FEEDBACK FROM PREVIOUS ITERATION
    The following feedback was generated from evaluating your previous attempt. Use it to guide your next refinement:
    
    Hybrid Automaton Specification v1:
```json
{
  "automaton": {
    "var": "x1",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] - 1.0 * x1[0]**3"
      }
    ],
    "edge": []
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "dim": 2,
    "need_reset": false,
    "non_linear_items": "x1[0]**3"
  }
}
```
Evaluation Results:
<lambda>() takes 1 positional argument but 2 were given
Hybrid Automaton Specification v2:
```json
{
  "automaton": {
    "var": "x1",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] - 1.0 * x1[0]**3"
      }
    ],
    "edge": []
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "dim": 2,
    "need_reset": false,
    "non_linear_items": "x1[0]**3"
  }
}
```
Evaluation Results:
<lambda>() takes 1 positional argument but 2 were given

    
    