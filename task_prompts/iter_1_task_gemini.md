# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
A Hybrid Automaton models a system with:
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each regime)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)

## Interaction Protocol (Mandatory Structure)
You must structure your response in the following sections:
1.  **Phase 1: Visual Analysis**: Observations from `hybrid_automaton_image_analysis`.
2.  **Phase 2: Quantitative Verification**: Python code execution to estimate parameters and verify switches.
3.  **Phase 3: Refinement Logic**: Explanation of the constructed modes and guards.
4.  **Phase 4: Validation**: Output from `validate_hybrid_automaton_specification`.
5.  **Final Answer**: The complete, valid JSON specification.

## Analysis Workflow

### Phase 1: Visual Hypothesis Generation (Qualitative)
**Tool**: `hybrid_automaton_image_analysis`
**Goal**: Formulate initial hypotheses about the system's structure and dynamics.
1.  **Analyze the Image**: Look at the trajectory shapes and relationships between variables.
2.  **Propose Hypotheses**:
    *   **Dynamics**:
        *   "Periodic/Wavy" -> Suggests trigonometric terms (sin, cos) or harmonic oscillator.
        *   "Decay/Growth" -> Suggests damping/unstable terms.
        *   "Straight Lines" -> Suggests constant velocity or simple linear dynamics.
    *   **Structure (Modes & Switching)**:
        *   "Sharp Kinks/Corners" -> Suggests **Mode Switches** (change in vector field).
        *   "Jumps/Discontinuities" -> Suggests **Resets** (instantaneous state change).
        *   "Smooth but Complex" -> Could be nonlinear or a smooth switch (hidden mode).
3.  **Deliverable**: A list of hypotheses (e.g., "H1: 2 modes. H2: Switch at x>0.")

### Phase 2: Data-Driven Verification & Refinement (Quantitative)
**Tool**: Managed Agent (`data_analysis_expert`) or Python Code Interpreter
**Goal**: Verify hypotheses and extract precise parameters using the `.npz` data.
1.  **Load Data**: Access the raw numerical data from the `.npz` file.
2.  **Segment & Fit**:
    *   *Refine Switch Points*: Use code to find the exact indices/values where behavior changes (e.g., `np.where(np.abs(np.diff(state)) > threshold)`).
    *   *Parameter Estimation*:
        *   **Tip**: Use `scipy.optimize.curve_fit` or `np.linalg.lstsq`.
        *   **Strategy**: Split the data based on your hypothesized guard (e.g., `mask = state[0] > 0`). Fit Model A to `data[mask]` and Model B to `data[~mask]`.
    *   *Check Residuals*: If a linear fit has high residuals that look "wavy" or "parabolic", introduce nonlinear terms (`x^2`, `sin(x)`).

### Phase 3: Specification Construction
**Goal**: Synthesize findings into the JSON format.
1.  **Define Modes**: Create a mode entry for each distinct behavior identified.
2.  **Define Transitions (Edges)**:
    *   Use the precise guard conditions found in Phase 2 (e.g., `x1 >= 0.5`).
    *   Define resets if "Jumps" were confirmed in Phase 2.
3.  **Define Equations**: Use the estimated parameters to write the ODEs.
    *   Ensure the equations are in **higher-order form** (e.g., `x1[2] = ...` for 2nd order).

### Phase 4: Final Validation
**Tool**: `validate_hybrid_automaton_specification`
**Goal**: Ensure the generated JSON is syntactically and semantically correct before submission.
**Constraint**: If the validation tool returns a fixed JSON, you **MUST** use that fixed version.

## HA Refinement Guidelines (Occam's Razor)

1. **Parsimony**: Start with **Linear** dynamics and **Fewest** modes. Only add complexity (non-linearity, extra modes) if the fit error is high.
2. **Mode Count Strategy**:
   - **Visual Evidence**: Sharp corners = Multiple Modes.
   - **Smooth but Hybrid**: If a single nonlinear ODE fails to fit, try 2 simpler linear modes.
   - **Template Expansion**: The template provides 1 mode. You must copy/paste to create Mode 2, Mode 3, etc., if needed.
3. **Guard Conditions**:
   - Prefer single-variable thresholds (e.g., `x1 >= 0`).
   - Avoid complex arithmetic guards unless necessary.
4. **Reset Maps**:
   - Default: `need_reset: false` (Continuous state evolution).
   - Only use `need_reset: true` if you see vertical jumps in the state-time plot.

## ⛔ Negative Constraints (Do NOT do this)
*   **Do NOT change variable names**: Use exactly `x1`, `u1` as provided.
*   **Do NOT use state-space splitting**: If the system is 2nd order, use `x1[2]`, NOT `x1` and `x2`.
*   **Do NOT guess coefficients**: You must calculate them using the provided data.
*   **Do NOT hallucinate inputs**: If `input` is empty, do not use `u1` in equations.

## Computational Resources
You have access to managed Code Agent(s): `['data_analysis_expert']`
Use them for numerical computations, curve fitting, or complex mathematical derivations.
You MUST use the managed agents to verify your analysis and hypotheses about the system from the hybrid_automaton_image_analysis tool.

## Code Execution Capability
You can use Python Code to execute programs, which may help with your task-solving process.

## Hybrid Automaton Specification Format (JSON Schema)

### 🎯 GOAL: PARSIMONIOUS SYSTEM IDENTIFICATION
Your objective is to identify the **simplest possible** Hybrid Automaton that explains the data.

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
          "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*(\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*$",
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
          "pattern": "^([a-zA-Z_][a-zA-Z0-9_]*(\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)?$",
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
                    "x1": [
                        0
                    ],
                    "x2": [
                        "-0.1 * x1[0]"
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


### ⚠️ CRITICAL: Variable Count is PRE-DEFINED ⚠️
The `var` and `input` fields in the template below are **already correctly set** based on the ground truth data.
- **DO NOT** add or remove variables!
- **DO NOT** convert to state-space form (e.g., splitting 1 variable into x1, x2)!
- For single-variable systems: use higher-order ODE notation (e.g., `x1[2] = ...` for 2nd-order)
- Focus on inferring the **equations** (`eq`), **modes**, and **edge conditions** only!

## Initial Hybrid Automaton Specification (v0)
The `var` and `input` fields are pre-filled. Your task is to refine the **equations** and **structure**:

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
        "dt": 0.001,
        "total_time": 10.0,
        "order": 2,
        "need_reset": false,
        "non_linear_items": ""
    }
}
```

## Your Task
Generate an improved HA specification (v1) that better matches the observed trajectory data.
- **Keep `var: "x1"` and `input: "u1"` exactly as shown!**
- Make sure the HA specification is valid and complete.
- Refine the HA specification to improve trajectory matching and reduce TC (Change-Point Error), Mean Difference, and Maximum Difference.

## Available Data
The following data sources are provided:
- **Trace visualizations**: ['<image_0>', '<image_1>', '<image_2>']
- **Raw data files**: ['<npz_0>', '<npz_1>', '<npz_2>']



You MUST use the `hybrid_automaton_image_analysis` tool to analyze the image.

**VALIDATION TOOL**: Before submitting your final answer, you MUST call the `validate_hybrid_automaton_specification` tool to check for syntax and semantic errors.
⚠️ **IMPORTANT**: If validation returns a FIXED specification, use the corrected version in your final answer!