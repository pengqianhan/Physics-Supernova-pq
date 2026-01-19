# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
A Hybrid Automaton models a system with:
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each regime)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)


## Quality Criteria
Your HA specification will be evaluated on:
- **Trajectory Matching**: Simulated output should closely follow ground truth data
- **Mode Detection Accuracy**: Correct identification of switching instants (TC (Change-Point Error) < 0.01s is good, <= 0.001s is excellent)
- **State Error Minimization**: Low Mean Difference (Mean Difference) and Maximum Difference (Max Difference) between predicted and actual states (Max Difference < 0.005 is good, < 0.0001 is excellent)
## Tool and Sub-Agents Resources:

You MUST use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.Before submitting your final answer, you MUST call the `validate_hybrid_automaton_specification` tool to check for syntax and semantic errors. If validation returns a FIXED specification, use the corrected version in your final answer!

**Sub-Agents Resources:** You have access to managed Code Agent(s): `['data_analysis_expert']`. The `['data_analysis_expert']` have access to the npz data files and can be used to analyze the data.
## Code Execution Capability
You can use Python Code to execute programs, which may help with your task-solving process.
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

## ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification (v0)
The `var` and `input` fields are pre-filled.

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
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "x1, x2"` and `input: ""` exactly as shown!**
- Make sure the HA specification is valid and complete.
- Refine the HA specification to improve trajectory matching and reduce TC (Change-Point Error), Mean Difference, and Maximum Difference.

## Available Data
The following data sources are provided:
- **Trace visualizations**: ['<image_0>', '<image_1>', '<image_2>'], use the `hybrid_automaton_image_analysis` tool to analyze the image if you want to obtain more detailed information about the system.
- **Raw data files**: ['<npz_0>', '<npz_1>', '<npz_2>']. If you want to use the npz data to analyze the system, you MUSTuse the `data_analysis_expert` agent to analyze the data. You can not analyze the npz data directly.

## ⚠️ FEEDBACK FROM PREVIOUS ITERATION
The following feedback was generated from evaluating your previous attempt. Use it to guide your next refinement:
    
    

## Previously Explored HA Specifications
The following specifications have been explored in previous iterations.
Performance is ranked from best (lowest error) to worst. Use these as inspiration.

--- Explored Specifications (Ranked) ---

### Rank 1 (Iteration 1)
**Error**: 2.046695
```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -1.0 * x1[0] - 0.05 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x2 >= 5.25",
        "reset": {
          "x2": [
            "-0.9 * x2[0]"
          ]
        }
      },
      {
        "direction": "1 -> 1",
        "condition": "x2 <= -5.25",
        "reset": {
          "x2": [
            "-0.9 * x2[0]"
          ]
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
**Metrics**: tc: 0.0000, max_diff: 2.0467, mean_diff: 0.6218

-----------------------------------------

## Most Recent Attempt (Iteration 2):
```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2, x2[1] = -10.0 * x1 - 0.5 * x2"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x1 == 0",
        "reset": {
          "x2": [
            "-0.9 * x2[0]"
          ]
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
Evaluation Results:
The HA specification is not valid. Please try to generate a valid specification. Here is the error message: name 'x2' is not defined
    
    