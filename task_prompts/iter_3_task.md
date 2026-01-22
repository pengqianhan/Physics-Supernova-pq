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

## Analyzing Evaluation Results (For Iterations 2+)
When feedback includes an `Evaluation Artifacts (JSON)` section, you can analyze the comparison plot:
1. Find the `artifacts[].placeholder` in the JSON (e.g., `<iter_image_1>`, `<iter_image_2>`), which are the placeholders for the comparison plot. <iter_image_1> is the comparison plot of the 1st iteration, <iter_image_2> is the comparison plot of the 2nd iteration, and so on.
2. Call `hybrid_automaton_image_analysis' to analyze the image if you want to obtain more detailed information about the simulated trajectory and the ground truth trajectories.
3. The `plot_summary` in the artifacts provides a text fallback.

## Feedback from 3th attempt
Use it to guide your next refinement:



## Previously Explored HA Specifications
The following specifications have been explored in previous iterations.
Performance is ranked from best (lowest error) to worst. Use these as inspiration.

--- Explored Specifications (Ranked) ---

### Rank 1 (Iteration 2)
**Error**: 21.314997
```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = -0.5 * x1[0] - x2[0], x2[1] = 0.5 * x2[0]"
      },
      {
        "id": 2,
        "eq": "x1[1] = -0.5 * x1[0] + x2[0], x2[1] = -0.5 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "x1 <= 0",
        "reset": {
          "x2": [
            "-0.9 * x2[0]"
          ]
        }
      },
      {
        "direction": "2 -> 1",
        "condition": "x1 > 0",
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
    "total_time": 10.0
  }
}
```
**Metrics**: tc: 0.0000, max_diff: 172.1624, mean_diff: 21.3150

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
        "eq": "x1[1] = -0.5 * x1[0] - x2[0], x2[1] = 0.5 * x2[0]"
      },
      {
        "id": 2,
        "eq": "x1[1] = -0.5 * x1[0] + x2[0], x2[1] = -0.5 * x2[0]"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 2",
        "condition": "x1 <= 0",
        "reset": {
          "x2": [
            "-0.9 * x2[0]"
          ]
        }
      },
      {
        "direction": "2 -> 1",
        "condition": "x1 > 0",
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
    "total_time": 10.0
  }
}
```

## Evaluation Artifacts (JSON)
Use the `hybrid_automaton_image_analysis` tool with placeholder `<iter_image_2>` to analyze the comparison plot.
```json
{
  "iteration": 2,
  "metrics": {
    "TC (Change-Point Error)": 0.0,
    "Max Difference": 172.16235758493184,
    "Mean Difference": 21.31499672654544
  },
  "plot_summary": "The HA simulation exhibits **severe divergence** from the ground truth, indicated by a very high `max_diff` (172.16) and a significant `mean_diff` (21.31).\n\n**Visual Analysis:**\n1.  **$x_1$ Trajectory:** The simulated $x_1$ (dashed line) shows exponential growth, diverging rapidly towards $+240$ by $t=10$s. The ground truth $x_1$ (solid line) remains near zero, suggesting the system is nearly stable or oscillatory around zero in the ground truth, which the model fails to capture.\n2.  **$x_2$ Trajectory:** The simulated $x_2$ (dashed line) shows exponential decay towards $-240$. The ground truth $x_2$ remains near zero initially, then exhibits a slow, near-linear decay, suggesting a different underlying dynamics or mode structure.\n3.  **Mode Switches:** The ground truth shows numerous, frequent mode changes (indicated by many `change_points`), while the simulation appears stuck in a single mode (Mode 1, as all change points are at $t=0$ and $t=10$s).\n\n**Specific Issues & Suggestions:**\n\n1.  **Dynamics Mismatch:** The continuous dynamics in Mode 1 ($\\dot{x}_1 = -0.5 x_1 - x_2$, $\\dot{x}_2 = 0.5 x_2$) lead to instability for $x_1$ when $x_2$ is small and positive, causing the observed divergence. **Suggestion:** Re-evaluate the ODE coefficients in both modes to match the observed slower, bounded behavior of the ground truth.\n2.  **Guard/Switching Error:** The simulation never triggers the edge conditions ($x_1 \\le 0$ or $x_1 > 0$) to switch between Mode 1 and Mode 2, as $x_1$ remains positive and grows. **Suggestion:** The guard conditions must be corrected to reflect the true switching logic, or the dynamics must be adjusted so that $x_1$ crosses zero as observed in the ground truth's frequent switching pattern.\n3.  **Reset Action:** The reset action on $x_2$ ($\\pm 0.9 x_2[0]$) is aggressive and likely contributes to the large state deviations upon switching (though no switches occur). **Suggestion:** Remove or significantly reduce the reset magnitude, as the ground truth suggests smooth transitions or small, bounded jumps.",
  "artifacts": [
    {
      "placeholder": "<iter_image_2>",
      "caption": "Overlay: ground truth (solid) vs simulated (dash-dot)"
    }
  ]
}
```


