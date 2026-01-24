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
        "eq": "x1[1] = 0.0010 * x2[0], x2[1] = -0.0570 * x1[0] - 0.0157 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x1 <= 0 and x2 < 0",
        "reset": {
          "x1": [
            "0"
          ],
          "x2": [
            "-0.9000 * x2[0] - 0.01"
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
    "Max Difference": 1.1968584796256005,
    "Mean Difference": 0.5731380531492432
  },
  "Evaluation Plot": {
    "Placeholder": "<iter_image_1>",
    "Summary": "The simulation exhibits **severe divergence** from the ground truth, characterized by a complete failure to capture the periodic, reset-driven behavior.\n\n**Fit Quality & Visual Patterns:**\nThe metrics ($\\text{max\\_diff} \\approx 1.2$, $\\text{mean\\_diff} \\approx 0.57$) are high, confirming poor fit. Visually, the simulated trajectory ($\\text{x1, x2}$ dashed lines) immediately settles to a fixed point near $(1.4, -1.6)$, while the ground truth shows strong oscillations with periodic, near-vertical resets (characteristic of a hybrid system). The simulated system is stuck in a single continuous mode (mode 1) with no observed mode switches.\n\n**Specific Issues:**\n1.  **Missing Resets/Guards:** The ground truth clearly shows state jumps (resets) occurring frequently. The simulation's $\\text{change\\_points}$ array shows only one segment (0 to 10001), indicating the guard condition for the edge is never met, or the dynamics are incorrect.\n2.  **Incorrect Dynamics:** The continuous dynamics in Mode 1 are likely inaccurate, as the system should be oscillating, not converging to a fixed point.\n3.  **Phase/Amplitude Mismatch:** The simulated trajectory does not match the amplitude or frequency of the ground truth oscillations.\n\n**Suggestions for HA Specification Improvement:**\n1.  **Verify Guard Conditions:** The guard $\\text{\"x1 <= 0 and x2 < 0\"}$ is likely incorrect or too restrictive. Analyze the ground truth state values at the observed reset times ($\\text{change\\_points}$) to define the correct guards that trigger the edge transition.\n2.  **Refine Continuous Dynamics (Mode 1):** The current linear ODEs ($\\dot{x}_1 = 0.001 x_2$, $\\dot{x}_2 = -0.057 x_1 - 0.0157 x_2$) do not produce sustained oscillations. These must be re-identified to match the observed sinusoidal behavior between resets.\n3.  **Review Reset Map:** The reset map ($\\text{x2} \\rightarrow -0.9 x_2[0] - 0.01$) should be checked against the magnitude of the jump observed in the ground truth data at mode switches."
  }
}
