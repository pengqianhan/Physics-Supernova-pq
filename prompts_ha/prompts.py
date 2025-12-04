# HA Specification Format Documentation (for LLM understanding)
HA_SPEC_DOCUMENTATION = """
## Hybrid Automaton Specification Format

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
- Mass-spring-damper (2nd order): `"x1[2] = -k*x1[0] - c*x1[1] + F"`
- Van der Pol (2nd order): `"x1[2] = mu*(1 - x1[0]**2)*x1[1] - x1[0]"`
- Two coupled 1st-order: `"x1[1] = a*x1[0] + b*x2[0], x2[1] = c*x1[0] + d*x2[0]"`

### NOTES ON CONDITION FORMAT
- Both "x1 > 0" and "x1[0] > 0" are supported in conditions (auto-converted internally)
- Prefer bare variable names (x1, x2) for cleaner specifications
"""

# Valid JSON template for HA specification (parsed by json.loads)
# Example: Bouncing ball with gravity (2 variables: position x1, velocity x2)
initial_ha_spec_prompt = """{
    "automaton": {
        "var": "x1, x2",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = -9.8 + u1"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {
                    "x1": [0],
                    "x2": ["-0.9 * x2[0]"]
                }
            }
        ]
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "dim": 1,
        "need_reset": true,
        "non_linear_items": ""
    }
}"""

# Alternative example: Single-mode Duffing oscillator (2nd-order ODE, 1 variable)
duffing_ha_spec_prompt = """{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] - 0.5 * x1[0]**3 + u1"
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
}"""

# Example: Two-mode thermostat system (heating/cooling)
thermostat_ha_spec_prompt = """{
    "automaton": {
        "var": "x1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = -0.1 * x1[0] + 5"
            },
            {
                "id": 2,
                "eq": "x1[1] = -0.1 * x1[0] - 3"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "x1 >= 25"
            },
            {
                "direction": "2 -> 1",
                "condition": "x1 <= 15"
            }
        ]
    },
    "config": {
        "dt": 0.01,
        "total_time": 20.0,
        "dim": 1,
        "need_reset": false,
        "non_linear_items": ""
    }
}"""


if __name__ == "__main__":
    print("=" * 60)
    print("HA SPECIFICATION DOCUMENTATION")
    print("=" * 60)
    print(HA_SPEC_DOCUMENTATION)
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Bouncing Ball (with reset)")
    print("=" * 60)
    print(initial_ha_spec_prompt)
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Duffing Oscillator (2nd-order, nonlinear)")
    print("=" * 60)
    print(duffing_ha_spec_prompt)
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Thermostat (two-mode switching)")
    print("=" * 60)
    print(thermostat_ha_spec_prompt)