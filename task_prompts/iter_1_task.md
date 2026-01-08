# HYBRID AUTOMATON SYSTEM IDENTIFICATION TASK

## Your Role
You are a control systems engineer specializing in **Hybrid Automaton (HA) system identification**. Your objective is to infer a mathematically precise HA model from observed trajectory data that accurately captures the underlying switched dynamical system behavior.

## Problem Context
1. **Discrete modes** (operating regimes with distinct continuous dynamics)
2. **Mode-specific ODEs** (differential equations governing each regime)
3. **Switching conditions** (guard predicates triggering mode transitions)
4. **Reset maps** (state updates upon mode transitions)

## Available Data
The following trace data visualizations are provided (reference images using placeholders: `<image_0>`, `<image_1>`, etc.):
- State variable trajectories over time
- Input signals (if applicable)
- Potential mode-switch indicators (discontinuities, slope changes)

## Analysis Workflow

### Phase 1: Visual Hypothesis Generation (Qualitative)
**Tool**: `hybrid_automaton_image_analysis`
**Goal**: Formulate initial hypotheses about the system's structure and dynamics.
1.  **Analyze the Image**: Look at the trajectory shapes and relationships between variables.
2.  **Propose Hypotheses**:
    *   **Dynamics**:
        *   "Periodic/Wavy" -> Suggests trigonometric terms (sin, cos).
        *   "Decay/Growth" -> Suggests damping/unstable terms (negative/positive feedback).
        *   "Straight Lines" -> Suggests constant velocity or simple linear dynamics.
    *   **Structure (Modes & Switching)**:
        *   "Sharp Kinks/Corners" -> Suggests **Mode Switches** (change in vector field).
        *   "Jumps/Discontinuities" -> Suggests **Resets** (instantaneous state change).
        *   "Smooth but Complex" -> Could be nonlinear or a smooth switch (hidden mode).
3.  **Output**: Write down your hypotheses (e.g., "H1: System has 2 modes. H2: Switch occurs when x1 hits top/bottom. H3: Mode 1 is a damped oscillator.")

### Phase 2: Data-Driven Verification & Refinement (Quantitative)
**Tool**: Managed Agent (Code) or Python Code Interpreter
**Goal**: Verify hypotheses and extract precise parameters using the `.npz` data.
1.  **Load Data**: Access the raw numerical data from the `.npz` file.
2.  **Verify & Refine Hypotheses**:
    *   *Refine Switch Points*: If Phase 1 suggested a switch at "peaks", use code to find the exact state values where this happens. Is it exactly x=1.0 or x=0.98?
    *   *Verify Dynamics*:
        *   If H3 was "damped oscillator", try to fit a standard damped harmonic oscillator model to the data segment.
        *   Check the **residuals**. If the fit is bad, revise the hypothesis (e.g., add a nonlinear term like x^3).
    *   *Detect Subtle Modes*: Use "Windowed Error Analysis" (sliding window fit) to find mode switches that are invisible to the eye (smooth transitions).
3.  **Parameter Estimation**: Perform regression (e.g., `scipy.optimize.curve_fit` or Least Squares) on the segmented data to get the exact ODE coefficients.

### Phase 3: Specification Construction
**Goal**: Synthesize findings into the JSON format.
1.  **Define Modes**: Create a mode entry for each distinct behavior identified.
2.  **Define Transitions (Edges)**:
    *   Use the precise guard conditions found in Phase 2 (e.g., `x1 >= 0.5`).
    *   Define resets if "Jumps" were confirmed in Phase 2.
3.  **Define Equations**: Use the estimated parameters to write the ODEs.

### Phase 4: Final Validation
**Tool**: `validate_hybrid_automaton_specification`
**Goal**: Ensure the generated JSON is syntactically and semantically correct before submission.

## HA Refinement Guidelines
Adopt a **Parsimonious Modeling Approach** (Occam's Razor) - favor simpler explanations unless the data demands otherwise:

1. **Symbolic Identification**:
   - **Hypothesize Structure First**: Determine the likely functional form of the equations *before* estimating parameters.

2. **Mode Count Strategy (Adaptive Complexity)**:
   - **Visual Evidence Rules**: If the visual analysis (Step 1) detects sharp changes or discontinuities, you **MUST** use multiple modes (2 or more).
   - **Smooth but Hybrid**: If the trajectory is smooth but complex (e.g., varying frequency or damping), prefer a **Switched System** (2+ modes) over a single overly-complex nonlinear equation.
   - **Do NOT Force Single Mode**: Do not attempt to fit a single smooth ODE to data that clearly changes behavior. It is better to have 2 simple linear modes than 1 complex failing non-linear mode.
   - **Template Expansion**: The provided template below is for **1 Mode**. If you detect N modes, you **MUST copy/paste** the mode object to create Mode 2, Mode 3, ... Mode N in your JSON.

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
- **Mode Detection Accuracy**: Correct identification of switching instants (TC (Change-Point Error) < 0.01s is good, <= 0.001s is excellent)
- **State Error Minimization**: Low Mean Difference (Mean Difference) and Maximum Difference (Max Difference) between predicted and actual states (Max Difference < 0.005 is good, < 0.0001 is excellent)

You MUST use the `hybrid_automaton_image_analysis` tool to analyze the image.
**VALIDATION TOOL**: Before submitting your final answer, you MUST call the `validate_hybrid_automaton_specification` tool to check for syntax and semantic errors.
⚠️ **IMPORTANT**: If validation returns a FIXED specification, use the corrected version in your final answer!

## Computational Resources
You have access to managed Code Agent(s): `['data_analysis_expert']`
Use them for numerical computations, curve fitting, or complex mathematical derivations.
You MUST use the managed agents to verify your analysis and hypotheses about the system from the hybrid_automaton_image_analysis tool.

## Code Execution Capability
You can use Python Code to execute programs, which may help with your task-solving process.
## Hybrid Automaton Specification Format (JSON Schema)

### 🎯 GOAL: PARSIMONIOUS SYSTEM IDENTIFICATION
Your objective is to identify the **simplest possible** Hybrid Automaton that explains the data.
- **Penalty**: You will be penalized for adding unnecessary modes or complex nonlinear terms.
- **Strategy**: Start with a single mode with linear dynamics. Only add complexity if error remains high.

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
        "total_time"
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
            "description": "Reset values for each derivative order. Use '' to preserve current value.",
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
- Make sure the HA specification is valid and complete.
- Refine the HA specification to improve trajectory matching and reduce TC (Change-Point Error), Mean Difference, and Maximum Difference.
