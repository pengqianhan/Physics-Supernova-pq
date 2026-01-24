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
- ODE notation:
   - `x[0]` = variable value
   - `x[1]` = first derivative (dx/dt)
   - `x[2]` = second derivative (d²x/dt²)
   - Left side = highest derivative (e.g., `x1[2] = ...` for order=2)
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)

## IMPORTANT: Python Boolean Syntax
When constructing the HA specification dictionary in Python code:
- Use `True`/`False` (Python syntax), **NOT** `true`/`false` (JSON syntax)
- Example: `"need_reset": True` not `"need_reset": true`
- Example: `"self_loop": False` not `"self_loop": false`

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
- **Raw data files**: If you want to use the npz data to analyze the system, you MUST use the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.

### NPZ Data Format (for reference - use via `data_analysis_expert` agent)
Each NPZ file contains:
- `state`: numpy array, shape `(num_variables, num_steps)` - state trajectories
  - Access: `x1 = data['state'][0, :]`, `x2 = data['state'][1, :]`
- `input`: numpy array, shape `(num_inputs, num_steps)` - input signals (if applicable)
  - Access: `u1 = data['input'][0, :]`

**WARNING**: Do NOT use placeholder names like `<npz_0>` as file paths! Use the `data_analysis_expert` agent which has access to the actual file paths.

"
## Previously Explored HA Specifications with Feedback",
            "Use these as inspiration to guide your next refinement.",
            "Use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section."
            
-----------------------------------------

The 2th attempt result:
  1. HA JSON Specification:
```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = -0.5 * x1[0] + x2[0], x2[1] = -0.5 * x2[0]"
      },
      {
        "id": 2,
        "eq": "x1[1] = -0.5 * x1[0] - x2[0], x2[1] = -0.5 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "x1 <= 0",
        "reset": {
          "x2": [
            "-0.94*x2[0]"
          ]
        }
      },
      {
        "direction": "2 -> 1",
        "condition": "x1 >= 0",
        "reset": {
          "x2": [
            "-0.94*x2[0]"
          ]
        }
      }
    ]
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "order": 1
  }
}
```
  2. Evaluation Feedback:
{
  "Evaluation Metrics": {
    "TC (Change-Point Error)": 0.863,
    "Max Difference": 1.1518219624636648,
    "Mean Difference": 0.3816216088514591
  },
  "Evaluation Plot": {
    "Placeholder": "<iter_image_2>",
    "Summary": "The HA simulation exhibits significant discrepancies with the ground truth, particularly concerning the fast, oscillatory behavior of $x_2$.\n\n**Fit Quality & Visual Patterns:**\nThe overall fit quality is poor ($\\text{tc}=0.863$, $\\text{max\\_diff}=1.15$). The simulation fails to capture the rapid, high-amplitude oscillations seen in the ground truth $x_2$. The simulated $x_2$ (light blue dashed line) is nearly constant or shows very slow drift, while the ground truth $x_2$ (light blue solid line) exhibits sharp, periodic drops. The simulation's $x_1$ (dark blue dashed line) shows a slow, damped trend, which loosely follows the envelope of the ground truth $x_1$, but lacks the high-frequency component.\n\n**Specific Issues:**\n1.  **Mode Switching Mismatch:** The ground truth exhibits frequent, rapid mode switches (indicated by the sharp drops in $x_2$), with approximately 10 switches in the first 5 seconds. The simulation only recorded **one** major switch at $t=0.863$s, followed by settling into Mode 2 until $t=10$s. This indicates the guard conditions are not being met correctly or the dynamics within the modes are wrong.\n2.  **$x_2$ Dynamics Failure:** The simulation's dynamics for $x_2$ are completely incorrect. The ground truth suggests $x_2$ is being reset or driven to large negative values periodically, which is not replicated by the simulated dynamics or resets.\n3.  **Reset Action:** The reset condition $\\text{reset}: x_2 \\leftarrow -0.94 x_2[0]$ seems insufficient or incorrectly timed to generate the observed behavior.\n\n**Suggestions for Improvement:**\n1.  **Re-evaluate Mode 1 & 2 ODEs:** The current linear dynamics ($x_1[1] = -0.5 x_1[0] \\pm x_2[0], x_2[1] = -0.5 x_2[0]$) are too simplistic to generate the observed high-frequency oscillations. The ground truth likely involves coupling terms that drive $x_2$ more aggressively.\n2.  **Correct Guard Conditions:** The guard conditions ($\\text{x1} \\le 0$ and $\\text{x1} \\ge 0$) are likely incorrect or incomplete. The sharp drops in $x_2$ suggest a threshold condition related to $x_2$ itself, or a time-based event, is missing or misidentified.\n3.  **Investigate Reset Dynamics:** The reset for $x_2$ must be significantly different or occur much more frequently. If the system is a relaxation oscillator, the reset might need to be a hard jump (e.g., $x_2 \\leftarrow \\text{constant}$) rather than a scaled version of the previous state, or the transition condition must be met almost instantly."
  }
}
