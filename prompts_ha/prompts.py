# HA Specification Format Documentation (for LLM understanding)
HA_SPEC_DOCUMENTATION = """
## Hybrid Automaton Specification Format

### AUTOMATON SECTION
- `var`: State variables (comma-separated for multiple, e.g., "x1, x2, x3")
- `input`: External input signals (comma-separated, e.g., "u1, u2"). Can be omitted if no input.
- `mode`: List of discrete operating modes, each containing:
  - `id`: Unique mode identifier (integer >= 1)
  - `eq`: Differential equation(s) using x[k] notation where k is derivative order
    * x1[0] = current value of x1
    * x1[1] = first derivative (dx1/dt)  
    * x1[2] = second derivative (d²x1/dt²)
    * Left-hand side: highest derivative; Right-hand side: expression
    * Multiple equations: comma-separated, e.g., "x1[1] = ..., x2[1] = ..."
    * Example 1st-order: "x1[1] = -2 * x1[0] + u1"
    * Example 2nd-order: "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"
- `edge`: List of discrete transitions, each containing:
  - `direction`: "source_mode -> target_mode" (e.g., "1 -> 2", "1 -> 1" for self-loop)
  - `condition`: Guard predicate triggering transition
    * **IMPORTANT**: Use BARE variable names (x1, x2), NOT indexed notation (x1[0])!
    * The condition is evaluated with current state values directly
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
- `dt`: Integration time step in seconds (smaller = more accurate, typical: 0.001 or 0.01)
- `total_time`: Total simulation duration in seconds
- `dim`: Order of the ODE system (1=first-order, 2=second-order, etc.)
- `need_reset`: Boolean to enable state resets on mode transitions (true if edges have reset maps)
- `non_linear_items`: Nonlinear/cross terms in dynamics for reference (e.g., "x1[0]**3", "x1[0]*x2[0]")
"""

# Valid Python dict template for HA specification (parseable by json.loads or eval)
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