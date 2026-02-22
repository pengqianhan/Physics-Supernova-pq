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
- `Max Difference < 0.002`: good fit
- `Mean Difference < 0.001`: good fit

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
        "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0",
                "reset": {
                    "x1": ["", "x[1] * 0.8"],
                }
            }
        ]
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
        "dt": 0.0001,
        "total_time": 2.0,
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
        "var": "x1",
        "input": "u1",
        "mode": [
            {
                "id": 1,
                "eq": "x1[2] = x1[1] + x1[0] + u1"
            }
        ],
        "edge": []
    },
    "config": {
        "dt":,
        "total_time":,
        "order": 2,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

## Your Task
Generate an improved HA specification that better matches the observed trajectory data.
- **Keep `var: "x1"` and `input: "u1"` exactly as shown!**
- Make sure the HA specification is valid and complete according to the JSON Schema.
- Refine the HA specification to improve trajectory matching and reduce `Max Difference`, `Mean Difference`.

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

### Detecting Mode Transitions via Numerical Analysis
For higher-order systems (order >= 2), position trajectories may appear smooth even when mode transitions occur, because resets often affect **derivatives** (velocity, acceleration) rather than position directly. Use the `data_analysis_expert` agent to:
1. **Compute numerical derivatives**: `velocity = np.diff(state, axis=1) / dt` and `acceleration = np.diff(velocity, axis=1) / dt`
2. **Detect discontinuities**: sudden jumps in velocity or acceleration indicate potential mode transition points and resets
3. **Segment-wise analysis**: once candidate transition points are identified, analyze each segment's dynamics separately to infer mode-specific ODEs and guard conditions

"
## Previously Explored HA Specifications with Feedback:
Use these as inspiration to guide your next refinement.
Use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section.
 If you want to check the fit performance of the HA specification, you can use the `hybrid_automaton_image_analysis` tool to analyze the comparison plot through the `Placeholder` in the `Evaluation Plot` section. For example, `hybrid_automaton_image_analysis(image_ref='<iter_image_2_0>', question='How is the fit performance of the HA specification?')` means to analyze the '2nd iteration, 0th file' comparison plot.`.

-----------------------------------------

The 1th attempt result:
  1. HA JSON Specification:
```json
{
  "automaton": {
    "var": "x1",
    "input": "u1",
    "mode": [
      {
        "id": 1,
        "eq": "x1[2] = -0.2 * x1[1] - 0.5 * x1[0] ** 3 + u1"
      },
      {
        "id": 2,
        "eq": "x1[2] = -0.54 * x1[1] - 1.5 * x1[0] ** 3 + u1"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "abs(x1) >= 1.2",
        "reset": {
          "x1[1]": [
            "0.95*x1[1]"
          ]
        }
      },
      {
        "direction": "2 -> 1",
        "condition": "abs(x1) <= 0.8",
        "reset": {
          "x1[1]": [
            "0.95*x1[1]"
          ]
        }
      }
    ]
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "order": 2
  }
}
```
  2. Evaluation Feedback:
{
  "Evaluation Metrics (Averaged)": {
    "Max Difference": 0.9382943236339885,
    "Mean Difference": 0.31368359980371213,
    "Num Ground Truth Files": 3
  },
  "Evaluation Plot": {
    "Placeholders": [
      "<iter_image_1_0>",
      "<iter_image_1_1>",
      "<iter_image_1_2>"
    ],
    "Summary": "The identification results show a **poor fit** characterized by significant **phase lead** and frequency mismatch. The simulated trajectory oscillates consistently faster than the ground truth, leading to a `max_diff` of ~0.94 and rapid divergence in phase.\n\n**Specific Issues:**\n1.  **Frequency Mismatch:** The simulated system's \"stiffness\" is too high. The cubic terms ($-0.5x_1^3$ and $-1.5x_1^3$) drive the state"
  },
  "Per-File Results": [
    {
      "ground_truth_index": 0,
      "ground_truth_file": "ground_truth_0.npz",
      "plot_placeholder": "<iter_image_1_0>",
      "max_diff": 0.8100110251062057,
      "mean_diff": 0.3249686635199039
    },
    {
      "ground_truth_index": 1,
      "ground_truth_file": "ground_truth_1.npz",
      "plot_placeholder": "<iter_image_1_1>",
      "max_diff": 1.0165515235952371,
      "mean_diff": 0.31988232698076347
    },
    {
      "ground_truth_index": 2,
      "ground_truth_file": "ground_truth_2.npz",
      "plot_placeholder": "<iter_image_1_2>",
      "max_diff": 0.9883204222005224,
      "mean_diff": 0.29619980891046893
    }
  ]
}
