# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data.

## Problem Context
A Hybrid Automaton models a system with:
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each mode)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)


## Metrics (lower is better)
- `Max Difference < 0.01`: good fit
- `Mean Difference < 0.005`: accurate overall
- `TC (Change-Point Error) < 0.01s`: mode switch timing correct

## Tool and Sub-Agents Resources:

You MUST use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.

You have access to managed Code Agent: `['data_analysis_expert']` to analyze the npz data files.
## Hybrid Automaton Specification Format (JSON Schema)

### JSON Schema Definition
Your output MUST conform to this JSON Schema:

```json
{
  "title": "Hybrid Automaton Specification",
  "description": "Schema for defining hybrid automaton systems with modes, transitions, and configuration",
  "type": "object",
  "required": [
    "automaton",
    "config"
  ],
  "additionalProperties": true,
  "properties": {
    "automaton": {
      "type": "object",
      "description": "The hybrid automaton definition with variables, modes, and transitions",
      "required": [
        "var",
        "mode",
        "edge"
      ],
      "properties": {
        "var": {
          "type": "string",
          "description": "Comma-separated list of state variable names (e.g., 'x1' or 'x1, x2')",
          "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\\s*,\\s*[a-zA-Z_][a-zA-Z0-9_]*)*$",
          "minLength": 1,
          "examples": [
            "x1",
            "x1, x2",
            "x, y, z"
          ]
        },
        "input": {
          "type": "string",
          "description": "Comma-separated list of input variable names (e.g., 'u1' or 'u1, u2'). Empty string if no inputs.",
          "pattern": "^([a-zA-Z_][a-zA-Z0-9_]*(\\s*,\\s*[a-zA-Z_][a-zA-Z0-9_]*)*)?$",
          "default": "",
          "examples": [
            "",
            "u1",
            "u1, u2"
          ]
        },
        "mode": {
          "type": "array",
          "description": "List of discrete modes (operating regimes) in the automaton",
          "minItems": 1,
          "items": {
            "$ref": "#/definitions/mode"
          }
        },
        "edge": {
          "type": "array",
          "description": "List of discrete transitions between modes (can be empty [])",
          "items": {
            "$ref": "#/definitions/edge"
          }
        }
      },
      "additionalProperties": false
    },
    "config": {
      "type": "object",
      "description": "Simulation and learning configuration parameters",
      "required": [
        "dt",
        "total_time",
        "order"
      ],
      "properties": {
        "dt": {
          "type": "number",
          "description": "Integration time step in seconds",
          "exclusiveMinimum": 0,
          "examples": [
            0.001,
            0.01
          ]
        },
        "total_time": {
          "type": "number",
          "description": "Total simulation duration in seconds",
          "exclusiveMinimum": 0,
          "examples": [
            10.0,
            20.0
          ]
        },
        "order": {
          "type": "integer",
          "description": "Order of the ODE system (1=first-order, 2=second-order, etc.)",
          "minimum": 1,
          "default": 1
        },
        "need_reset": {
          "type": "boolean",
          "description": "Whether variable resets occur on mode transitions, if there are more than 2 modes, the reset is required",
          "default": false
        },
        "non_linear_items": {
          "type": "string",
          "description": "Nonlinear/cross terms in dynamics (e.g., 'x1[?]**3', 'x1[?]*x2[?]') in ODEs, which is eq in mode. The nonlinear/cross terms MUST be consistent with the 'eq' in mode.",
          "default": ""
        },
        "self_loop": {
          "type": "boolean",
          "description": "Whether self-loop transitions are allowed",
          "default": false
        },
        "need_bias": {
          "type": "boolean",
          "description": "Whether to include constant term in ODEs"
        }
      },
      "additionalProperties": true
    }
  },
  "definitions": {
    "mode": {
      "type": "object",
      "description": "A discrete mode (operating regime) with its continuous dynamics",
      "required": [
        "id",
        "eq"
      ],
      "properties": {
        "id": {
          "type": "integer",
          "description": "Unique mode identifier (must be >= 1)",
          "minimum": 1
        },
        "eq": {
          "type": "string",
          "description": "ODE equation(s) defining the continuous dynamics. Format: 'var[order] = expression'. Multiple equations separated by commas.",
          "minLength": 1,
          "examples": [
            "x1[1] = -2 * x1[0] + u1",
            "x1[2] = x1[1] - x1[0] ** 2 + u1"
          ]
        }
      },
      "additionalProperties": false
    },
    "edge": {
      "type": "object",
      "description": "A discrete transition (edge) between modes",
      "required": [
        "direction",
        "condition"
      ],
      "properties": {
        "direction": {
          "type": "string",
          "description": "Transition direction in format 'source -> target' (e.g., '1 -> 2')",
          "pattern": "^\\d+\\s*->\\s*\\d+$"
        },
        "condition": {
          "type": "string",
          "description": "Guard condition that triggers the transition. Use bare variable names (x1, not x1[0]).",
          "minLength": 1,
          "examples": [
            "x1 > 0.5",
            "x1 <= 0 and x2 > 1",
            "abs(x1) >= 1.2"
          ]
        },
        "reset": {
          "type": "object",
          "description": "Optional state reset map. Keys are variable names, values are arrays of reset expressions.",
          "additionalProperties": {
            "type": "array",
            "description": "Reset values for each derivative order: [x[0], x[1], ...]. Use '' to preserve current value.",
            "items": {
              "oneOf": [
                {
                  "type": "string"
                },
                {
                  "type": "number"
                }
              ]
            }
          },
          "examples": [
            {
              "x1": [
                0
              ],
              "x2": [
                "-0.9 * x2[0]"
              ]
            },
            {
              "x": [
                "",
                "x[1] * 0.95"
              ]
            }
          ]
        }
      },
      "additionalProperties": false
    }
  }
}
```


### Examples

**Single-mode 2nd-order:**
```json
{
    "automaton": {
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = x1[1] + x1[0] + x1[0] ** 2 + u1"
            }
        ],
        "edge": []
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 2,
        "need_reset": false,
        "non_linear_items": "x1[?] ** 2"
    }
}
```

**Two-mode switching:**
```json
{
    "automaton": {
        "var": "x1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x1[0] + 5"
            },
            {
                "id": 2,
                "eq": "x1[1] = -x1[0] - 3"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "x1 >= 4"
            },
            {
                "direction": "2 -> 1",
                "condition": "x1 <= 9"
            }
        ]
    },
    "config": {
        "dt": 0.01,
        "total_time": 20.0,
        "order": 1,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

**With reset map:**
```json
{
    "automaton": {
        "var": "x1, x2",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = x1[0] + u1"
            }
        ],
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {
                    "x1": [0],
                    "x2": ["-0.1 * x1[0]"]
                }
            }
        ]
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 1,
        "need_reset": true,
        "non_linear_items": ""
    }
}
```


## CRITICAL: Variable Count is PRE-DEFINED
- The `var` and `input` fields in  Initial Hybrid Automaton Specification (v0) are **already correctly set** based on the ground truth data. **DO NOT** add or remove variables!
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- ODEnotation:
   - `x[0]` = variable value
   - `x[1]` = first derivative (dx/dt)
   - `x[2]` = second derivative (d²x/dt²)
   - Left side = highest derivative (e.g., `x1[2] = ...` for order=2)
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)

## Initial Hybrid Automaton Specification (v0)

```json
{
    "automaton": {
        "var": "x1, x2",
        "input": "",
        "mode": [
            {
                "id": 1,
                "eq": "x1[1] = -0.5 * x1[0] + x2[0], x2[1] = -0.5 * x2[0]"
            }
        ],
        "edge": []
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 1,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

## Your Task
Generate an improved HA specification that better matches the observed trajectory data.
- **Keep `var: "x1, x2"` and `input: ""` exactly as shown!**
- Make sure the HA specification is valid and complete according to the JSON Schema.
- Refine the HA specification to improve trajectory matching and reduce `Max Difference`, `Mean Difference`, and `TC (Change-Point Error)`.

## Available Data
- **Trace visualizations**: ['<image_0>', '<image_1>', '<image_2>'], use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.
- **Raw data files**: ['<npz_0>', '<npz_1>', '<npz_2>']. If you want to use the npz data to analyze the system, you MUST use the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.

"
## Previously Explored HA Specifications with Feedback",
            "Use these as inspiration to guide your next refinement.",
            "Use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section."
            
-----------------------------------------

The 1th attempt result:
  1. HA JSON Specification:
```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -160 * x1[0] - 0.4 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x1 <= 0.1",
        "reset": {
          "x2": [
            "-0.8*x2[0]"
          ]
        }
      }
    ]
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "self_loop": true
  }
}
```
  2. Evaluation Feedback:
{
  "Evaluation Metrics": {
    "TC (Change-Point Error)": 0.0,
    "Max Difference": 3.0129008061859257,
    "Mean Difference": 0.26181261960002605
  },
  "Evaluation Plot": {
    "Placeholder": "<iter_image_1>",
    "Summary": "## HA System Identification Summary\n\n**Overall Fit Quality:** Poor, characterized by significant amplitude and phase discrepancies, especially in the initial transient phase. The Mean Difference ($\\mu=0.26$) is acceptable for the steady-state, but the Max Difference ($\\text{Max}=3.01$) indicates severe initial errors.\n\n**Visual Patterns:**\n1.  **State $x_2$ (Light Blue):** The simulated trajectory exhibits highly exaggerated, periodic oscillations with much larger amplitudes than the ground truth, particularly for $t < 2s$. This indicates the simulation is overshooting the true dynamics significantly after each mode switch.\n2.  **Mode Switching:** The simulated jumps (sawtooth pattern) occur much more frequently and with larger magnitude changes than the ground truth, suggesting the guard condition or reset map is triggering incorrectly or too aggressively.\n3.  **Steady State:** Both trajectories converge to approximately $(0, 0)$, but the convergence rate of the simulation appears slower or more oscillatory than the ground truth in the later stages.\n\n**Specific Issues:**\n*   **Amplitude Error:** The reset map for $x_2$ ($\\text{reset}: x_2 = -0.8 x_2[0]$) is likely too strong or the guard condition is met at inappropriate times, leading to the large initial overshoots in $x_2$.\n*   **Phase/Timing Error:** The frequency of the simulated jumps does not match the ground truth's behavior, suggesting the guard condition ($\\text{guard}: x_1 \\le 0.1$) is not accurately capturing the physical switching logic.\n\n**Suggestions for HA Specification Improvement:**\n\n1.  **Refine Reset Map:** The reset factor for $x_2$ (currently $-0.8$) is too aggressive. Test a smaller magnitude (e.g., $-0.2$ to $-0.5$) to dampen the overshoot immediately following a switch.\n2.  **Adjust Guard Condition:** The guard $x_1 \\le 0.1$ seems too broad or misplaced. Analyze the ground truth data to find the precise state value (or a combination of states) that triggers the switch, potentially incorporating $x_2$ into the guard condition for better timing.\n3.  **Review ODEs:** The continuous dynamics ($x_1' = x_2$, $x_2' = -160 x_1 - 0.4 x_2$) might be slightly inaccurate, contributing to the overall divergence, although the primary issue appears to be the discrete transitions. Consider parameter estimation refinement for the damping term ($0.4$) or stiffness ($160$)."
  }
}
