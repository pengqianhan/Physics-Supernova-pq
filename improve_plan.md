# Improvement Plan: Skeleton-First + Numerical Fitting for HA Coefficient Estimation

## 1. Problem Diagnosis

### Evidence from Experiments

| | Ground Truth | Gemini3Flash Best (iter 5) | Gemini3Pro Best (iter 9) |
|---|---|---|---|
| **mean_diff** | 0.0 | 0.161 | 0.201 |
| **Mode count** | 2 | 2 | 3 (over-segmented) |
| **Skeleton** | `c1*u + c2*x[1] + c3*x[0] + c4*x[0]**3` | Same ✅ | Same ✅ |
| **Mode 1 damping** | -0.5 | -1.25 (2.5x off) | -0.87 (1.7x off) |
| **Mode 1 cubic** | -1.5 | -9.32 (6.2x off) | +54.31 (sign flip!) |
| **Mode 2 damping** | -0.2 | -0.50 (2.5x off) | -1.16 (5.8x off) |

**Root Cause**: LLMs correctly identify ODE structure (skeleton) but fail at precise numerical coefficient estimation because:

1. **LLMs are language models, not numerical optimizers** — they produce "plausible-looking" numbers, not mathematically optimal ones
2. **Chaotic sensitivity of nonlinear ODEs** — small coefficient errors in `x[0]**3` cause exponential trajectory divergence
3. **Compensating local minima** — the LLM inflates one coefficient to compensate for another, creating physically meaningless parameter combinations (e.g., cubic=54.31 with linear=-12.75)
4. **No gradient signal** — the LLM receives only aggregate metrics (mean_diff) as feedback, not per-parameter gradients

### What Already Works

The existing `optimize_ha_params.py` (Nevergrad-based optimizer) demonstrates that:
- Given a correct skeleton, Nevergrad *can* optimize coefficients
- But it plateaued at mean_diff ≈ 0.15 with the duffing system due to:
  - LLM-assigned bounds that were too narrow initially
  - Coupling between ODE coefficients and guard/reset parameters creates a rugged loss landscape
  - Single global optimization over ALL parameters simultaneously

---

## 2. Evaluation of Proposed Strategy

### User's Proposal Summary
> Separate skeleton generation from coefficient estimation. First generate a mode.eq skeleton, then fit coefficients numerically against trace data using a segment-by-segment approach.

### Verdict: **Approved with Refinements**

The core insight is sound — decomposing the problem into (1) structural identification and (2) parameter estimation is a well-established approach in system identification literature (e.g., NARMAX, SINDy). The LLM excels at the structural part; numerical optimization handles the quantitative part.

### Strengths of the Proposal ✅

1. **Plays to each tool's strength**: LLM → structure/skeleton; optimizer → coefficients
2. **Segment-by-segment fitting reduces coupling**: Fitting one mode's ODE on its own segment isolates parameters, making the optimization landscape smoother
3. **Progressive validation**: Checking if a single-segment fit generalizes to other segments is an elegant way to detect whether modes share the same dynamics
4. **Early skeleton rejection**: If fitting fails on the first segment, the skeleton itself is wrong — no need to waste iterations on coefficient tuning

### Weaknesses and Needed Refinements ⚠️

1. **Segment boundary detection needs formalization**: "Observe if there's obvious segmentation from the trace plot" is too vague for an automated pipeline. Need a programmatic change-point detection step, not just visual inspection.

2. **Fitting method not specified**: `scipy.optimize.curve_fit` assumes analytical Jacobian availability and works best for explicit functions. For ODE systems, we need an **ODE-aware fitting method** — specifically, minimize the trajectory error by integrating the ODE with candidate parameters and comparing against data (shooting method).

3. **Guard/reset parameters are excluded**: The proposal focuses on mode.eq coefficients but ignores edge conditions and reset maps, which are equally important. The Duffing system's guard thresholds (0.8, 1.2) and reset coefficients (0.95) also need estimation.

4. **Multi-mode fitting order matters**: The proposal suggests "fit first segment, then check others" but doesn't handle the case where the first segment's mode is short or noisy. Should fit the **longest** segment first for robustness.

5. **Skeleton validation criterion needs quantification**: "If fitting error is small" needs a concrete threshold.

---

## 3. Refined Strategy: Two-Phase HA Identification

### Phase 1: Skeleton Generation (LLM-driven)

The LLM's job is ONLY to produce:
1. **Number of modes** (integer)
2. **ODE skeleton for each mode** — symbolic form with placeholder coefficients, e.g., `x[2] = c1*u + c2*x[1] + c3*x[0] + c4*x[0]**3`
3. **Edge structure** — which transitions exist (e.g., 1→2, 2→1) and the general form of guard conditions (e.g., `abs(x) <= threshold` vs `x >= threshold`)
4. **Reset structure** — general form (e.g., `x[1] * coefficient`)

The LLM should **NOT** specify any numerical coefficient values. All numbers become placeholders.

#### Prompt Changes for Phase 1

Add the following instruction block to the task prompt:

```
## IMPORTANT: Skeleton-Only Output Mode

Your primary task is to identify the STRUCTURE of the hybrid automaton, NOT the exact coefficients.

For ODE equations, use PLACEHOLDER coefficients:
  GOOD: "x[2] = c1 * u + c2 * x[1] + c3 * x[0] + c4 * x[0] ** 3"
  BAD:  "x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3"

For guard conditions, use PLACEHOLDER thresholds:
  GOOD: "abs(x) <= threshold_1"
  BAD:  "abs(x) <= 0.8"

For reset maps, use PLACEHOLDER coefficients:
  GOOD: "x[1] * reset_coeff_1"
  BAD:  "x[1] * 0.95"

Focus your analysis on:
1. How many distinct dynamical modes exist? (Look for qualitative behavior changes)
2. What is the functional form of each mode's ODE? (Linear? Cubic nonlinearity? Etc.)
3. What triggers mode switches? (State thresholds? Time-based?)
4. Are there resets at transitions? (Velocity jumps? Position resets?)
```

### Phase 2: Segment-by-Segment Numerical Fitting

Once the skeleton is fixed, optimize coefficients numerically using the following algorithm:

#### Step 2.1: Programmatic Change-Point Detection

Instead of relying on visual inspection, use programmatic methods:

```python
def detect_segments(npz_path: str) -> List[Dict]:
    """
    Detect segments from trace data using ground-truth change points
    (if available in npz) or automated detection.

    Returns list of dicts:
      [{'start': int, 'end': int, 'mode': int, 'duration': float}, ...]
    """
    data = np.load(npz_path)

    # If change_points exist in data (ground truth), use them directly
    if 'change_points' in data and 'mode' in data:
        cp = data['change_points']
        mode = data['mode']
        segments = []
        for i in range(len(cp) - 1):
            start, end = int(cp[i]), int(cp[i+1])
            seg_mode = int(mode[start])
            segments.append({
                'start': start, 'end': end,
                'mode': seg_mode,
                'duration': (end - start) * dt
            })
        return segments

    # Fallback: Use derivative-based change-point detection
    # (e.g., PELT algorithm from ruptures library, or
    #  second-derivative discontinuity detection)
    state = data['state']
    # ... automated detection logic ...
```

**Key insight**: The training data (`sample_*.npz`) may not contain `change_points` and `mode` arrays, but the LLM can observe segmentation from the trajectory plots. The fitting algorithm should support both cases:
- **With mode labels**: directly use known segments
- **Without mode labels**: the LLM identifies approximate segment boundaries from visual analysis, and numerical refinement optimizes the exact boundaries

#### Step 2.2: Per-Segment ODE Fitting (Single Shooting)

For each segment, solve the coefficient estimation as an optimization problem:

```python
def fit_segment_coefficients(
    skeleton: str,           # e.g., "x[2] = {c1}*u + {c2}*x[1] + {c3}*x[0] + {c4}*x[0]**3"
    state_data: np.ndarray,  # shape: (num_steps_in_segment,) — observed trajectory
    input_data: np.ndarray,  # shape: (num_steps_in_segment,) — input signal
    dt: float,
    initial_state: np.ndarray  # [x(0), x'(0)] for 2nd-order
) -> Tuple[np.ndarray, float]:
    """
    Fit coefficients by minimizing trajectory error using single shooting.

    Method:
    1. Parameterize skeleton with candidate coefficients
    2. Integrate ODE from initial_state using scipy.integrate.solve_ivp
    3. Compare simulated trajectory to observed state_data
    4. Minimize MSE using scipy.optimize.minimize (L-BFGS-B)

    Returns: (optimal_coefficients, fitting_error)
    """
```

**Why single shooting, not collocation?**
- Single shooting directly mimics how `HAEvaluator` computes trajectories
- This ensures the fitting objective is identical to the evaluation metric
- For segments of moderate length (100-2000 steps), single shooting is efficient
- If segments are very long (>5000 steps), consider multiple shooting to avoid gradient vanishing

**Why `scipy.optimize.minimize` instead of Nevergrad?**
- For per-segment fitting with ~4 coefficients, the optimization landscape is usually smooth (within a single mode, there's no switching discontinuity)
- Gradient-based methods (L-BFGS-B) converge much faster than gradient-free methods for smooth problems
- Use numerical gradients (finite differences) since analytical Jacobians of ODE solutions are complex
- Fallback to Nevergrad only if L-BFGS-B fails (detected by high residual or convergence failure)

#### Step 2.3: Segment-by-Segment Fitting Algorithm

```
Algorithm: PROGRESSIVE_SEGMENT_FIT

Input:
  - skeleton: ODE skeleton with placeholder coefficients
  - segments: List of detected segments (sorted by duration, longest first)
  - state_data, input_data: Full trajectory arrays
  - dt: time step

Output:
  - mode_coefficients: Dict[mode_id → coefficients]
  - guard_thresholds: Dict[edge → threshold]
  - reset_coefficients: Dict[edge → coefficient]

Procedure:

1. SORT segments by duration (longest first)

2. FIT the longest segment:
   coeffs_1, error_1 = fit_segment_coefficients(skeleton, segment_1_data, ...)

   IF error_1 > SKELETON_ERROR_THRESHOLD (e.g., 0.05):
     → Skeleton may be wrong. Return failure signal to LLM for skeleton revision.

3. FOR each remaining segment (by decreasing length):
   a. SIMULATE with coeffs_1 on this segment
   b. COMPUTE simulation error

   IF error < MODE_MATCH_THRESHOLD (e.g., 0.02):
     → This segment uses the same mode. Assign coeffs_1.

   ELSE:
     c. FIT new coefficients for this segment
     new_coeffs, new_error = fit_segment_coefficients(skeleton, segment_data, ...)

     IF new_error < SKELETON_ERROR_THRESHOLD:
       → Same skeleton, different coefficients → NEW MODE discovered
       → Check if new_coeffs is close to any previously discovered mode
          (use relative L2 distance of coefficient vectors)
       IF close to existing mode:
         → Merge (average coefficients or keep the one with lower error)
       ELSE:
         → Register as new mode

     ELSE:
       → Even fitting fails → This segment may need a DIFFERENT skeleton
       → Flag for LLM revision

4. DETERMINE guard thresholds:
   For each transition between modes:
   - Identify the state value at the transition point (from change_points)
   - Average over all transitions of the same type (e.g., all 1→2 transitions)
   - Use these averaged values as initial estimates
   - Refine with bounded optimization (±20% of the averaged value)

5. DETERMINE reset coefficients:
   For each transition with a reset:
   - Compare state just before vs. just after the transition
   - For velocity resets: reset_coeff = x'(after) / x'(before)
   - Average over all transitions of the same type
   - Refine with bounded optimization

6. GLOBAL REFINEMENT (optional, if budget allows):
   - Take the assembled full HA specification
   - Run Nevergrad optimization with NARROW bounds (±20% of segment-fitted values)
   - This fine-tunes the inter-mode coupling (guards + resets affect ODE integration across modes)
```

#### Step 2.4: Error Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| `SKELETON_ERROR_THRESHOLD` | 0.05 | If per-segment fitting error exceeds this, the skeleton is likely wrong |
| `MODE_MATCH_THRESHOLD` | 0.02 | If simulating with mode A's coefficients on segment B yields error < this, B is also mode A |
| `MODE_MERGE_THRESHOLD` | 0.15 (relative L2) | If two coefficient vectors are within 15% relative distance, merge them as the same mode |

These thresholds should be configurable and may need tuning per-system.

---

## 4. Integration with Existing Architecture

### Where This Fits in the Pipeline

```
Current pipeline:
  LLM → full HA spec (skeleton + coefficients) → HAEvaluator → feedback → LLM refinement
                        ↑ PROBLEM: coefficients are wrong

Proposed pipeline:
  LLM → skeleton-only HA spec → NUMERICAL FITTER → full HA spec → HAEvaluator → feedback
                                     ↑ NEW COMPONENT
  If fitting fails:
    → feedback to LLM: "skeleton validation failed on segment X, error=Y, suggest new skeleton"
```

### Implementation Plan

#### New File: `utils/ha_segment_fitter.py`

Core component that implements the segment-by-segment fitting algorithm:

```python
class HASegmentFitter:
    """
    Fits ODE coefficients to trajectory data segment-by-segment.

    Usage:
        fitter = HASegmentFitter(
            skeleton_spec=skeleton_ha_dict,  # HA spec with placeholder coefficients
            npz_paths=["data_all/.../ground_truth_0.npz"],
            dt=0.001,
            total_time=10.0
        )
        result = fitter.fit()
        # result.ha_spec — complete HA with fitted coefficients
        # result.per_segment_errors — fitting quality per segment
        # result.mode_map — which segments belong to which mode
    """
```

Key methods:
- `detect_segments()` — programmatic segmentation
- `fit_single_segment(skeleton, segment_data)` — L-BFGS-B coefficient fitting
- `classify_segments(fitted_modes)` — assign segments to modes
- `estimate_guards(transitions)` — determine guard thresholds
- `estimate_resets(transitions)` — determine reset coefficients
- `global_refinement(assembled_spec)` — optional Nevergrad fine-tuning
- `fit()` — orchestrates the full pipeline

#### Modifications to `run_llm_ha_gamma.py`

1. **Prompt modification**: Add skeleton-only instruction to `obtain_task_and_images()`
2. **Post-LLM processing**: After agent returns skeleton, call `HASegmentFitter`
3. **Feedback enrichment**: Include per-segment fitting results in feedback to LLM

#### Modifications to `optimize_ha_params.py`

The existing Nevergrad optimizer becomes the "global refinement" step (Step 6 above), with much narrower bounds since segment fitting provides good initial estimates.

### Modified Iteration Loop

```python
for iteration in range(1, max_iterations + 1):
    # Phase 1: LLM generates skeleton
    result = managerAgent.run(task, images=compressed_trace_images)
    skeleton_spec = extract_skeleton(result)

    # Phase 2: Numerical fitting
    fitter = HASegmentFitter(skeleton_spec, npz_paths, dt, total_time)
    fit_result = fitter.fit()

    if fit_result.success:
        # Use the numerically-fitted HA spec for evaluation
        ha_spec = fit_result.ha_spec

        # Optional: global refinement with Nevergrad (narrow bounds)
        if fit_result.max_segment_error > REFINEMENT_THRESHOLD:
            refined = optimize_ha_parameters_generic(
                ha_spec, input_data_path,
                # Narrow bounds: ±20% of fitted values
            )
            ha_spec = refined['ha_spec']
    else:
        # Fitting failed → skeleton is wrong
        # Feed back fitting diagnostics to LLM
        feedback += f"Skeleton fitting failed: {fit_result.diagnostics}"
        continue

    # Evaluate
    success, metrics, feedback_str, _ = evaluate_ha_specification_with_feedback(...)
```

---

## 5. Expected Impact

### Duffing System Projection

With the proposed approach:

1. **Skeleton identification** (LLM): Already works — both Gemini3Flash and Gemini3Pro correctly identify `x[2] = c1*u + c2*x[1] + c3*x[0] + c4*x[0]**3`

2. **Segment detection**: Ground truth has 12 change points, creating ~12 segments. Longest segments are ~2000 steps (mode 1, steps 529→1256) — plenty of data for fitting 4 coefficients.

3. **Per-segment fitting**: With L-BFGS-B on a 4-parameter, smooth, single-mode ODE:
   - Expected convergence in <100 function evaluations per segment
   - Each evaluation is a single ODE integration (~0.01s)
   - Total per-segment fitting: ~1s

4. **Mode discovery**: Fitting all mode-1 segments should yield consistent coefficients (damping≈0.5, linear≈1.0, cubic≈1.5). Mode-2 segments should yield different but consistent coefficients (damping≈0.2, linear≈1.0, cubic≈0.5).

5. **Guard estimation**: Transition state values at 1→2 transitions should cluster around x≈0.8, and 2→1 transitions around x≈1.2.

6. **Projected mean_diff**: < 0.01 (vs current best of 0.161), approaching the ground truth accuracy.

### Comparison with Current Approaches

| Approach | Iterations Needed | mean_diff | Coefficient Accuracy |
|---|---|---|---|
| Current LLM-only (50 iters) | 50 | ~0.16 | Poor (5-60x off) |
| Current Nevergrad (optimize_ha_params.py) | 1 (but 1000+ evals) | ~0.15 | Moderate |
| **Proposed skeleton+fitting** | **1-3** | **<0.01 (projected)** | **High** |

### Key Advantages

1. **Dramatically fewer LLM iterations**: The LLM only needs to get the skeleton right (which it already does in 1-3 iterations), not the coefficients
2. **Deterministic coefficient estimation**: Numerical fitting is reproducible; LLM coefficient guessing is not
3. **Diagnostic feedback**: If fitting fails, we know *exactly* which segment and why — much richer feedback than aggregate mean_diff
4. **Scales to harder systems**: For systems with many modes or high-order ODEs, the per-segment approach decomposes a hard global problem into easy local problems

---

## 6. Potential Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Segment fitting gets stuck in local minimum | Medium | Use multi-start: run L-BFGS-B from 5 random initial points per segment |
| Change-point detection is inaccurate on training data (no labels) | High | Use LLM's visual analysis to provide approximate boundaries, then refine numerically |
| Skeleton is wrong but fits well on some segments | Low | Cross-validate: a correct skeleton should fit well on ALL segments of its mode |
| Global refinement doesn't converge | Medium | Keep segment-fitted values as fallback; only use global refinement if it improves |
| Guards depend on derivatives (not just state values) | Medium | Extend guard estimation to include velocity conditions: check both x and x[1] at transitions |

---

## 7. Implementation Priority

### Phase A (Core — implement first)
1. `utils/ha_segment_fitter.py` — segment detection + per-segment ODE fitting
2. Prompt modification for skeleton-only output
3. Integration in `run_llm_ha_gamma.py` main loop

### Phase B (Refinement)
4. Guard and reset estimation from transition data
5. Mode merging logic
6. Global refinement with narrow-bound Nevergrad

### Phase C (Robustness)
7. Multi-start for segment fitting
8. Automated change-point detection (for data without labels)
9. Skeleton validation heuristics (residual analysis)

---

## 8. Summary

**The user's core insight is correct and well-motivated**: separating skeleton identification (LLM's strength) from coefficient estimation (numerical optimizer's strength) is the right decomposition. The segment-by-segment fitting strategy is sound in principle.

**Key refinements added in this plan**:
1. **Programmatic change-point detection** instead of visual-only
2. **Single-shooting ODE fitting** (L-BFGS-B) instead of generic curve_fit
3. **Longest-segment-first ordering** for robustness
4. **Mode merging logic** to handle over-segmentation
5. **Guard/reset estimation from transition data** (not just mode.eq)
6. **Global refinement** as optional final step with narrow bounds
7. **Quantified thresholds** for skeleton validation, mode matching, and mode merging
