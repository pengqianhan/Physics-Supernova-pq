"""
Combined SINDy + Nevergrad Hybrid Automaton Parameter Optimization

Pipeline:
  1. SINDy Agent: Fits mode ODEs (mode.eq) from trajectory data using PySINDy
  2. Nevergrad Agent: Optimizes edge parameters (edge.condition & edge.reset)
     using gradient-free optimization against ground truth

Usage:
    python optimize_sindy_nevergrad.py  # Runs duffing oscillator demo
"""

import hashlib
import json
import math
import os
import re
import sys
import functools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ

import nevergrad as ng
from loky import ProcessPoolExecutor

from pydantic import BaseModel, Field

try:
    import litellm
    from dotenv import load_dotenv
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/Dainarx_code'))
from HA_evaluation import HAEvaluator


# ============================================================================
# Pydantic Schemas for LLM Edge Estimation
# ============================================================================

class EdgeBound(BaseModel):
    """A single estimated edge parameter with bounds."""
    name: str = Field(..., description="Descriptive name, e.g., 'guard_threshold_1_to_2', 'reset_coef_x1'")
    value: float = Field(..., description="Estimated numeric value based on transition statistics")
    lower_bound: float = Field(..., description="Suggested lower bound for Nevergrad optimization")
    upper_bound: float = Field(..., description="Suggested upper bound for Nevergrad optimization")
    reasoning: str = Field(..., description="Brief explanation of how this value was estimated")


class EdgeEstimationResult(BaseModel):
    """Result of LLM-based edge parameter estimation."""
    updated_edges: str = Field(..., description="JSON string of the updated edges list with estimated numeric values")
    bounds: List[EdgeBound] = Field(..., description="List of estimated edge parameters with bounds, one per numeric value in edges")


# ============================================================================
# SINDy Agent — Fit mode ODEs from trajectory data
# ============================================================================

def _segment_by_mode(
    state: np.ndarray,
    inp: np.ndarray,
    mode: np.ndarray,
    change_points: np.ndarray,
    dt: float,
    order: int,
    min_segment_len: int = 50,
) -> Dict[int, List[Dict[str, np.ndarray]]]:
    """
    Segment trajectory data by mode and compute derivatives.

    Returns:
        Dict mapping mode_id -> list of segment dicts, each with keys:
          'X': (n_points, n_features) array  [x, dx, ..., u]
          'X_dot': (n_points, n_targets) array  [highest derivative]
    """
    n_vars = state.shape[0]
    segments: Dict[int, List] = {}

    # Build list of (start, end) from change_points
    boundaries = []
    for i in range(len(change_points) - 1):
        s, e = int(change_points[i]), int(change_points[i + 1])
        e = min(e, state.shape[1])
        if e - s >= min_segment_len:
            boundaries.append((s, e))
    # Handle case where last change_point is not a sentinel
    if len(change_points) > 0:
        last = int(change_points[-1])
        if last < state.shape[1] and state.shape[1] - last >= min_segment_len:
            boundaries.append((last, state.shape[1]))

    for s, e in boundaries:
        mid = mode[s]
        seg_state = state[:, s:e]   # (n_vars, seg_len)
        seg_inp = inp[:, s:e] if inp.ndim == 2 else inp[s:e].reshape(1, -1)

        # Compute derivatives up to `order` for each variable
        derivs = []  # list of arrays, each (seg_len,)
        for v in range(n_vars):
            x_v = seg_state[v]
            derivs.append(x_v)  # 0th derivative = value
            d = x_v.copy()
            for _ in range(order):
                d = np.gradient(d, dt)
                derivs.append(d)

        # X = [x0, dx0, ..., x1, dx1, ..., u0, u1, ...]
        # X_dot = [highest derivative for each var]
        features = []
        targets = []
        for v in range(n_vars):
            base = v * (order + 1)
            # Features: derivatives 0..(order-1)
            for k in range(order):
                features.append(derivs[base + k])
            # Target: the order-th derivative
            targets.append(derivs[base + order])

        # Append input columns as features
        for ui in range(seg_inp.shape[0]):
            features.append(seg_inp[ui])

        # Trim edges to avoid finite-difference artifacts
        trim = max(order, 2)
        X = np.column_stack(features)[trim:-trim]
        X_dot = np.column_stack(targets)[trim:-trim]

        segments.setdefault(mid, []).append({'X': X, 'X_dot': X_dot})

    return segments


def _build_feature_names(var_names: List[str], order: int, input_names: List[str]) -> List[str]:
    """Build feature column names matching _segment_by_mode output."""
    names = []
    for v in var_names:
        for k in range(order):
            names.append(f"{v}[{k}]")
    names.extend(input_names)
    return names


def _sindy_equation_to_ha_format(
    model: ps.SINDy,
    var_names: List[str],
    input_names: List[str],
    order: int,
    target_idx: int,
    threshold: float = 1e-6,
) -> str:
    """
    Convert a SINDy model equation to HA JSON equation format.

    Example output: "x[2] = -0.3 * x[1] + 1.0 * x[0] - 1.0 * x[0]**3 + u"
    """
    coefs = model.coefficients()[target_idx]  # 1D array
    feature_names = model.get_feature_names()

    lhs_var = var_names[target_idx] if len(var_names) > 1 else var_names[0]
    lhs = f"{lhs_var}[{order}]"

    terms = []
    for coef, fname in zip(coefs, feature_names):
        if abs(coef) < threshold:
            continue

        # Convert SINDy feature name to HA format
        ha_fname = _convert_feature_name(fname, var_names, input_names, order)

        if ha_fname == "1":  # bias/constant term
            terms.append(f"{coef:.6g}")
        elif abs(coef - 1.0) < 1e-8:
            terms.append(ha_fname)
        elif abs(coef + 1.0) < 1e-8:
            terms.append(f"-{ha_fname}")
        else:
            terms.append(f"{coef:.6g} * {ha_fname}")

    rhs = " + ".join(terms) if terms else "0"
    # Clean up "+-" -> "-"
    rhs = rhs.replace("+ -", "- ")
    return f"{lhs} = {rhs}"


def _convert_feature_name(
    fname: str, _var_names: List[str], input_names: List[str], _order: int
) -> str:
    """
    Convert a PySINDy feature name to HA equation format.

    SINDy names like 'x0[0]', 'x0[0]^2', 'x0[0] x0[1]' ->
    HA names like 'x[0]', 'x[0] ** 2', 'x[0] * x[1]'
    """
    # Handle constant/bias term
    if fname.strip() == "1":
        return "1"

    # Split product terms (SINDy uses space as multiplication)
    parts = fname.split(" ")
    converted = []
    for part in parts:
        # Check for power: e.g., "x0[0]^3"
        power = ""
        if "^" in part:
            part, power_str = part.rsplit("^", 1)
            power = f" ** {power_str}"

        # Check if it's an input variable name
        matched = False
        for inp_name in input_names:
            if part == inp_name:
                converted.append(inp_name + power)
                matched = True
                break

        if not matched:
            # It's a state variable feature: map column name back
            # Feature names from _build_feature_names: "x[0]", "x[1]", "x2[0]", etc.
            converted.append(part + power)

        if not matched and not any(c in part for c in ['[', ']']):
            # Fallback: keep as-is
            pass

    return " * ".join(converted) if len(converted) > 1 else (converted[0] if converted else fname)


def sindy_fit_modes(
    ha_spec: Dict[str, Any],
    npz_file_paths: List[str],
    poly_degree: int = 3,
    sindy_threshold: float = 0.05,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    SINDy Agent: Fit mode ODEs from trajectory data.

    Args:
        ha_spec: Initial HA specification (provides structure: var names, modes, order)
        npz_file_paths: Paths to ground truth .npz files
        poly_degree: Max polynomial degree for SINDy feature library
        sindy_threshold: Sparsity threshold for STLSQ optimizer
        verbose: Print progress

    Returns:
        Updated HA spec with SINDy-fitted mode equations
    """
    if verbose:
        print("=" * 60)
        print("[SINDy Agent] Fitting mode ODEs from trajectory data")
        print("=" * 60)

    automaton = ha_spec['automaton']
    config = ha_spec.get('config', {})
    dt = config.get('dt', 0.001)
    order = config.get('order', 2)

    # Parse variable and input names
    var_str = automaton.get('var', 'x')
    input_str = automaton.get('input', '')
    var_names = [v.strip() for v in var_str.split(',') if v.strip()]
    input_names = [v.strip() for v in input_str.split(',') if v.strip()] if input_str else []
    n_vars = len(var_names)

    mode_ids = sorted([m['id'] for m in automaton['mode']])
    if verbose:
        print(f"  Variables: {var_names}, Inputs: {input_names}, Order: {order}")
        print(f"  Modes to fit: {mode_ids}")

    # Collect segments from all NPZ files
    all_segments: Dict[int, List] = {}
    for npz_path in npz_file_paths:
        data = np.load(npz_path)
        state = data['state']  # (n_vars, n_steps)
        inp = data['input']
        if inp.ndim == 1:
            inp = inp.reshape(1, -1)
        mode_arr = data['mode']
        cp = data['change_points']

        segs = _segment_by_mode(state, inp, mode_arr, cp, dt, order)
        for mid, seg_list in segs.items():
            all_segments.setdefault(mid, []).extend(seg_list)

    if verbose:
        for mid in sorted(all_segments):
            total_pts = sum(s['X'].shape[0] for s in all_segments[mid])
            print(f"  Mode {mid}: {len(all_segments[mid])} segments, {total_pts} total points")

    # Build feature names for SINDy
    feat_names = _build_feature_names(var_names, order, input_names)
    if verbose:
        print(f"  Feature columns: {feat_names}")

    # Build feature library
    poly_lib = PolynomialLibrary(degree=poly_degree, include_interaction=True, include_bias=True)
    feature_library = poly_lib  # Start simple; extend if needed

    # Fit SINDy per mode
    updated_spec = json.loads(json.dumps(ha_spec))  # deep copy
    for mode_entry in updated_spec['automaton']['mode']:
        mid = mode_entry['id']
        if mid not in all_segments or not all_segments[mid]:
            if verbose:
                print(f"  [Mode {mid}] No data segments found, keeping original equation")
            continue

        # Concatenate all segments for this mode
        X_all = np.vstack([s['X'] for s in all_segments[mid]])
        X_dot_all = np.vstack([s['X_dot'] for s in all_segments[mid]])

        if verbose:
            print(f"\n  [Mode {mid}] Fitting SINDy on {X_all.shape[0]} points, {X_all.shape[1]} features")

        # Fit SINDy model
        optimizer = STLSQ(threshold=sindy_threshold)
        model = ps.SINDy(
            feature_library=feature_library,
            optimizer=optimizer,
        )

        try:
            model.fit(X_all, x_dot=X_dot_all, t=dt, feature_names=feat_names)
        except Exception as e:
            if verbose:
                print(f"  [Mode {mid}] SINDy fit failed: {e}, keeping original")
            continue

        if verbose:
            print(f"  [Mode {mid}] SINDy model:")
            model.print()

        # Convert each target (one per state variable) to HA equation format
        eq_parts = []
        for t_idx in range(n_vars):
            eq_str = _sindy_equation_to_ha_format(
                model, var_names, input_names, order, t_idx
            )
            eq_parts.append(eq_str)

        mode_entry['eq'] = ", ".join(eq_parts)
        if verbose:
            print(f"  [Mode {mid}] HA equation: {mode_entry['eq']}")

    return updated_spec


# ============================================================================
# Transition Statistics Collection
# ============================================================================

def collect_transition_statistics(
    ha_spec: Dict[str, Any],
    npz_file_paths: List[str],
) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Collect transition statistics from change points in trajectory data.

    For each change point, records state values and derivatives just before
    and after the transition. This data can be used to estimate guard thresholds
    and reset coefficients.

    Returns:
        Dict mapping (from_mode, to_mode) -> list of observation dicts.
        Each observation dict has keys like '{var}_vals_before' and '{var}_vals_after',
        each containing a list of [value, 1st_deriv, ..., nth_deriv].
    """
    automaton = ha_spec['automaton']
    config = ha_spec.get('config', {})
    dt = config.get('dt', 0.001)
    order = config.get('order', 2)

    var_str = automaton.get('var', 'x')
    var_names = [v.strip() for v in var_str.split(',') if v.strip()]
    n_vars = len(var_names)

    transition_data: Dict[Tuple[int, int], List[Dict]] = {}

    for npz_path in npz_file_paths:
        data = np.load(npz_path)
        state = data['state']   # (n_vars, n_steps)
        mode = data['mode']     # (n_steps,)
        cp = data['change_points']

        # Compute derivatives for each variable up to `order`
        all_derivs = []  # all_derivs[var][k] = k-th derivative array
        for v in range(n_vars):
            derivs = [state[v]]
            d = state[v].copy()
            for _ in range(order):
                d = np.gradient(d, dt)
                derivs.append(d)
            all_derivs.append(derivs)

        # Analyze each change point
        for i in range(len(cp) - 1):
            cp_idx = int(cp[i + 1])
            if cp_idx >= len(mode) or cp_idx < 1:
                continue

            m_before = int(mode[cp_idx - 1])
            m_after = int(mode[min(cp_idx, len(mode) - 1)])
            if m_before == m_after:
                continue

            entry = {}
            for v in range(n_vars):
                vn = var_names[v]
                entry[f'{vn}_vals_before'] = [all_derivs[v][k][cp_idx - 1] for k in range(order + 1)]
                entry[f'{vn}_vals_after'] = [all_derivs[v][k][min(cp_idx, len(mode) - 1)] for k in range(order + 1)]

            transition_data.setdefault((m_before, m_after), []).append(entry)

    return transition_data


# ============================================================================
# LLM-Based Edge Parameter Estimation
# ============================================================================

def format_transition_stats_for_llm(
    transition_data: Dict[Tuple[int, int], List[Dict]],
    var_names: List[str],
    order: int,
) -> str:
    """
    Format transition statistics into a readable summary for the LLM prompt.

    For each transition direction, shows:
      - State values at transition: median, std, range
      - Derivative ratios (after/before): median, std
      - Number of observations
    """
    lines = []
    for (from_m, to_m), entries in sorted(transition_data.items()):
        lines.append(f"Transition {from_m} -> {to_m} ({len(entries)} observations):")

        for vn in var_names:
            lines.append(f"  Variable '{vn}':")
            for k in range(order + 1):
                label = f"{vn}[{k}]" if k > 0 else vn
                # Collect before-values for this derivative order
                before_vals = [e[f'{vn}_vals_before'][k] for e in entries
                               if f'{vn}_vals_before' in e and k < len(e[f'{vn}_vals_before'])]
                after_vals = [e[f'{vn}_vals_after'][k] for e in entries
                              if f'{vn}_vals_after' in e and k < len(e[f'{vn}_vals_after'])]

                if before_vals:
                    bv = np.array(before_vals)
                    lines.append(f"    {label} before transition: median={np.median(bv):.6f}, "
                                 f"std={np.std(bv):.6f}, range=[{np.min(bv):.6f}, {np.max(bv):.6f}]")
                if after_vals:
                    av = np.array(after_vals)
                    lines.append(f"    {label} after transition:  median={np.median(av):.6f}, "
                                 f"std={np.std(av):.6f}, range=[{np.min(av):.6f}, {np.max(av):.6f}]")

                # Derivative ratios (after/before)
                if before_vals and after_vals:
                    ratios = []
                    for bv_i, av_i in zip(before_vals, after_vals):
                        if abs(bv_i) > 1e-8:
                            ratios.append(av_i / bv_i)
                    if ratios:
                        rv = np.array(ratios)
                        lines.append(f"    {label} ratio (after/before): median={np.median(rv):.6f}, "
                                     f"std={np.std(rv):.6f}")

        lines.append("")

    return "\n".join(lines)


def _get_edge_estimation_prompt() -> str:
    """System prompt for LLM-based edge parameter estimation."""
    return """You are an expert at analyzing Hybrid Automaton (HA) transition dynamics.

Your task: Given an HA specification and transition statistics observed from ground truth data,
estimate the numeric values for edge parameters (guard conditions and reset coefficients).

TRANSITION STATISTICS INTERPRETATION:
- "before transition" values = state/derivative values just before the guard fires
- "after transition" values = state/derivative values just after the reset is applied
- "ratio (after/before)" = how derivatives change across the transition (indicates reset coefficients)

GUARD CONDITION ESTIMATION:
- The guard threshold should be near the MEDIAN of state values at which transitions fire
- For conditions like "abs(x) <= threshold", use the median of |x| before transition
- For conditions like "x >= threshold", use the median of x before transition
- If the standard deviation is high, the guard may involve multiple variables or be complex

RESET COEFFICIENT ESTIMATION:
- Reset coefficients multiply state/derivative values: "x[k] * coef"
- Estimate coef from the MEDIAN derivative ratio (after/before) at the corresponding order
- For velocity resets (e.g., "x[1] * coef"), use the x[1] ratio
- Restitution coefficients are typically in [0.01, 2.0]

BOUNDS ESTIMATION:
- Wide bounds when std is high relative to median (uncertain estimate)
- Narrow bounds when std is low (confident estimate)
- Guard thresholds: [median - 3*std, median + 3*std], minimum width of 20% of value
- Reset coefficients: always at least [0.01, 2.0] (physically motivated range)
- General: never make bounds narrower than +/-20% of the estimated value

OUTPUT FORMAT:
- updated_edges: A JSON string containing the edges list with your estimated numeric values
  replacing the original ones. Maintain the exact same structure (direction, condition format,
  reset format), only change the numeric values.
- bounds: One EdgeBound per numeric parameter in the edges, in the order they appear
  (first all condition parameters left-to-right, then all reset parameters left-to-right,
  for each edge in order).

IMPORTANT: Only change numeric values. Do NOT change the structural form of conditions
or resets (e.g., don't change "abs(x) <= N" to "x >= N")."""


# ============================================================================
# Edge Estimation Caching
# ============================================================================

def _get_edge_estimation_cache_path(
    ha_spec: Dict[str, Any],
    transition_stats_str: str,
    cache_dir: str = ".edge_estimation_cache",
) -> str:
    """Get cache file path based on MD5 of spec + transition stats."""
    combined = json.dumps(ha_spec, sort_keys=True) + transition_stats_str
    cache_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"edge_estimation_{cache_hash}.json")


def _save_edge_estimation_cache(
    result: Dict[str, Any],
    cache_path: str,
    verbose: bool = True,
) -> None:
    """Save edge estimation result to a local JSON cache file."""
    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"[Cache] Saved edge estimation to {cache_path}")


def _load_edge_estimation_cache(
    cache_path: str,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load edge estimation result from cache if it exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        if verbose:
            print(f"[Cache] Loaded edge estimation from {cache_path}")
        return data
    except Exception as e:
        if verbose:
            print(f"[Cache] Failed to load cache ({e}), will re-estimate with LLM")
        return None


# ============================================================================
# LLM Edge Estimation Function
# ============================================================================

def estimate_edge_params_with_llm(
    ha_spec: Dict[str, Any],
    npz_file_paths: List[str],
    model_id: str = "gemini/gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    timeout: int = 60,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[List[Dict]]]:
    """
    Use LLM to estimate edge parameters from transition statistics.

    The LLM receives the HA spec and observed transition statistics, then
    estimates guard thresholds and reset coefficients with informed bounds.

    Args:
        ha_spec: HA specification dict
        npz_file_paths: Paths to ground truth NPZ files
        model_id: LiteLLM model identifier
        api_key: API key (reads from GEMINI_API_KEY env var if None)
        timeout: API call timeout in seconds
        verbose: Print progress

    Returns:
        Tuple of (updated_ha_spec, list_of_bound_dicts_or_None).
        On failure, returns the original spec unchanged (deep copy) with None bounds,
        so Nevergrad still works with default bounds.
    """
    if not HAS_LITELLM:
        if verbose:
            print("[LLM Edge] litellm not available, returning original spec unchanged")
        return json.loads(json.dumps(ha_spec)), None

    # Load API key
    if api_key is None:
        load_dotenv(override=True)
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        if verbose:
            print("[LLM Edge] No GEMINI_API_KEY found, returning original spec unchanged")
        return json.loads(json.dumps(ha_spec)), None

    try:
        if verbose:
            print("\n" + "=" * 60)
            print("[LLM Edge] Estimating edge parameters with LLM")
            print("=" * 60)

        # Step 1: Collect transition statistics
        automaton = ha_spec['automaton']
        var_str = automaton.get('var', 'x')
        var_names = [v.strip() for v in var_str.split(',') if v.strip()]
        order = ha_spec.get('config', {}).get('order', 2)

        transition_data = collect_transition_statistics(ha_spec, npz_file_paths)

        if not transition_data:
            if verbose:
                print("  No transitions found in data, returning original spec unchanged")
            return json.loads(json.dumps(ha_spec)), None

        if verbose:
            for key, entries in sorted(transition_data.items()):
                print(f"  Transition {key[0]}->{key[1]}: {len(entries)} observations")

        # Step 2: Format stats for LLM
        stats_str = format_transition_stats_for_llm(transition_data, var_names, order)

        # Step 3: Check cache
        cache_path = _get_edge_estimation_cache_path(ha_spec, stats_str)
        cached = _load_edge_estimation_cache(cache_path, verbose)
        if cached is not None:
            updated_spec = json.loads(json.dumps(ha_spec))
            updated_spec['automaton']['edge'] = cached['edges']
            bounds = cached.get('bounds')
            return updated_spec, bounds

        # Step 4: Build prompt and call LLM
        ha_spec_str = json.dumps(ha_spec, indent=2)
        user_prompt = f"""Analyze this Hybrid Automaton specification and the observed transition statistics
to estimate edge parameter values (guard thresholds and reset coefficients).

HA Specification:
```json
{ha_spec_str}
```

Observed Transition Statistics:
```
{stats_str}
```

Estimate the numeric values for all edge conditions and resets based on the statistics above.
Provide updated edges with your estimates and suggested optimization bounds."""

        if verbose:
            print("\n[LLM Edge] Calling LLM for edge estimation...")

        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "system", "content": _get_edge_estimation_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            api_key=api_key,
            timeout=timeout,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "EdgeEstimationResult",
                    "schema": EdgeEstimationResult.model_json_schema(),
                    "strict": True,
                }
            }
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty response")

        result = EdgeEstimationResult.model_validate_json(content)

        # Step 5: Parse and apply estimated values
        try:
            estimated_edges = json.loads(result.updated_edges)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid edges JSON: {e}")

        # Validate edge count matches
        original_edges = ha_spec['automaton'].get('edge', [])
        if len(estimated_edges) != len(original_edges):
            raise ValueError(
                f"Edge count mismatch: LLM returned {len(estimated_edges)}, "
                f"expected {len(original_edges)}"
            )

        # Build updated spec
        updated_spec = json.loads(json.dumps(ha_spec))
        updated_spec['automaton']['edge'] = estimated_edges

        # Convert bounds to list of dicts
        bounds_list = [
            {
                'name': b.name,
                'value': b.value,
                'lower': b.lower_bound,
                'upper': b.upper_bound,
                'reasoning': b.reasoning,
            }
            for b in result.bounds
        ]

        if verbose:
            print(f"[LLM Edge] Estimated {len(bounds_list)} edge parameters:")
            for b in bounds_list:
                print(f"    {b['name']}: {b['value']:.6f} [{b['lower']:.4f}, {b['upper']:.4f}] — {b['reasoning']}")
            print(f"\n[LLM Edge] Updated edges:")
            for edge in estimated_edges:
                print(f"    {edge.get('direction')}: condition='{edge.get('condition')}', reset={edge.get('reset')}")

        # Step 6: Save to cache
        _save_edge_estimation_cache(
            {'edges': estimated_edges, 'bounds': bounds_list},
            cache_path, verbose
        )

        return updated_spec, bounds_list

    except Exception as e:
        if verbose:
            print(f"[LLM Edge] Failed: {type(e).__name__}: {e}")
            print("[LLM Edge] Returning original spec unchanged")
        return json.loads(json.dumps(ha_spec)), None


# ============================================================================
# Nevergrad Agent — Optimize edge conditions and resets
# ============================================================================

def _extract_edge_parameters(ha_spec: Dict[str, Any], verbose: bool = True) -> Tuple[List[Dict], str]:
    """
    Extract numeric parameters from edge conditions and resets only.

    Returns:
        Tuple of (parameter list, parameterized spec JSON string)
    """
    spec = json.loads(json.dumps(ha_spec))  # deep copy
    params = []
    idx = 0

    for edge in spec['automaton'].get('edge', []):
        # Parameterize condition
        if 'condition' in edge:
            cond = edge['condition']
            new_cond, cond_params, idx = _parameterize_expression(cond, idx, 'edge_condition')
            edge['condition'] = new_cond
            params.extend(cond_params)

        # Parameterize reset
        if 'reset' in edge:
            for var_name, reset_val in edge['reset'].items():
                if isinstance(reset_val, list):
                    new_list = []
                    for item in reset_val:
                        if isinstance(item, str) and item.strip():
                            new_item, item_params, idx = _parameterize_expression(
                                item, idx, 'edge_reset'
                            )
                            new_list.append(new_item)
                            params.extend(item_params)
                        else:
                            new_list.append(item)
                    edge['reset'][var_name] = new_list
                elif isinstance(reset_val, str) and reset_val.strip():
                    new_val, val_params, idx = _parameterize_expression(
                        reset_val, idx, 'edge_reset'
                    )
                    edge['reset'][var_name] = new_val
                    params.extend(val_params)

    parameterized_str = json.dumps(spec)
    if verbose:
        print(f"  Extracted {len(params)} edge parameters:")
        for p in params:
            print(f"    {p['name']}: {p['value']} [{p['lower']}, {p['upper']}]")

    return params, parameterized_str


def _parameterize_expression(
    expr: str, start_idx: int, location: str
) -> Tuple[str, List[Dict], int]:
    """
    Replace numeric literals in an expression with {param_N} placeholders.

    Skips:
      - Variable indices in brackets like x[0], x1[2]
      - Exponents after ** operator
    """
    params = []
    idx = start_idx

    # Regex: match numeric literals NOT inside brackets or after **
    # Strategy: process token by token
    result = []
    i = 0
    while i < len(expr):
        # Skip whitespace
        if expr[i].isspace():
            result.append(expr[i])
            i += 1
            continue

        # Skip variable references like x[0], x1[2]
        if expr[i:i+1].isalpha() or expr[i] == '_':
            # Read identifier
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            ident = expr[i:j]
            result.append(ident)
            i = j
            # Skip bracket index if present
            if i < len(expr) and expr[i] == '[':
                k = expr.index(']', i) + 1
                result.append(expr[i:k])
                i = k
            continue

        # Skip ** operator and its exponent
        if expr[i:i+2] == '**':
            result.append('**')
            i += 2
            # Skip whitespace
            while i < len(expr) and expr[i].isspace():
                result.append(expr[i])
                i += 1
            # Read exponent (number)
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            result.append(expr[i:j])
            i = j
            continue

        # Check for numeric literal (possibly negative sign already handled by operator)
        if expr[i].isdigit() or (expr[i] == '.' and i + 1 < len(expr) and expr[i+1].isdigit()):
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            # Check for scientific notation
            if j < len(expr) and expr[j] in ('e', 'E'):
                j += 1
                if j < len(expr) and expr[j] in ('+', '-'):
                    j += 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1

            num_str = expr[i:j]
            value = float(num_str)

            # Compute bounds — tight range around data-estimated values
            abs_val = abs(value) if value != 0 else 1.0
            margin = 0.3  # ±30% around the estimated value
            lower = max(0.001, abs_val * (1.0 - margin))
            upper = abs_val * (1.0 + margin)

            placeholder = f"{{param_{idx}}}"
            params.append({
                'name': f'edge_param_{idx}',
                'value': value,
                'lower': lower,
                'upper': upper,
                'location': location,
            })
            result.append(placeholder)
            idx += 1
            i = j
            continue

        # Other characters (operators, parentheses, etc.)
        result.append(expr[i])
        i += 1

    return ''.join(result), params, idx


def _reconstruct_from_template(template_str: str, params: np.ndarray) -> Dict[str, Any]:
    """Fill {param_N} placeholders with concrete values."""
    concrete = template_str
    for i in range(len(params)):
        concrete = concrete.replace(f"{{param_{i}}}", str(params[i]))
    return json.loads(concrete)


def _nevergrad_edge_objective(
    *,
    template_str: str,
    npz_file_paths: List[str],
    dt: float,
    total_time: float,
    verbose: bool,
    param_count: int,
    **kwargs
) -> float:
    """Top-level objective for Nevergrad (multiprocessing-safe)."""
    params = np.array([kwargs[f"param_{i}"] for i in range(param_count)])
    try:
        ha_spec = _reconstruct_from_template(template_str, params)

        mean_diffs = []
        for npz_path in npz_file_paths:
            evaluator = HAEvaluator(
                ha_dict=ha_spec,
                npz_file_path=npz_path,
                dt=dt,
                total_time=total_time
            )
            evaluator.load_ground_truth()
            evaluator.simulate()
            metrics = evaluator.compute_metrics()
            md = metrics.get('mean_diff', float('inf'))
            mean_diffs.append(md if md is not None else float('inf'))

        avg = float(np.mean(mean_diffs)) if mean_diffs else float('inf')
        if verbose:
            ps = ", ".join([f"{p:.4f}" for p in params])
            print(f"  edge_params=[{ps}] -> mean_diff={avg:.6f}")
        return avg

    except Exception as e:
        if verbose:
            print(f"  edge_params ERROR: {e}")
        return float('inf')


def nevergrad_optimize_edges(
    ha_spec: Dict[str, Any],
    npz_file_paths: List[str],
    budget: Optional[int] = None,
    num_workers: Optional[int] = None,
    llm_bounds: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Nevergrad Agent: Optimize edge condition and reset parameters.

    Args:
        ha_spec: HA spec (with SINDy-fitted mode equations)
        npz_file_paths: Ground truth NPZ file paths
        budget: Optimization budget (auto-calculated if None)
        num_workers: Parallel workers (auto if None)
        llm_bounds: Optional LLM-suggested bounds (list of dicts with 'value', 'lower', 'upper').
                    When provided and count matches extracted edge params, overrides the
                    default +/-30% bounds with LLM-informed bounds.
        verbose: Print progress

    Returns:
        Dict with 'ha_spec', 'error', 'edge_params'
    """
    if verbose:
        print("\n" + "=" * 60)
        print("[Nevergrad Agent] Optimizing edge parameters")
        print("=" * 60)

    dt = ha_spec.get('config', {}).get('dt', 0.001)
    total_time = ha_spec.get('config', {}).get('total_time', 10.0)

    # Extract edge parameters
    edge_params, template_str = _extract_edge_parameters(ha_spec, verbose=verbose)

    if not edge_params:
        if verbose:
            print("  No edge parameters to optimize.")
        return {'ha_spec': ha_spec, 'error': None, 'edge_params': {}}

    # Override bounds with LLM-suggested bounds if available and count matches
    if llm_bounds is not None and len(llm_bounds) == len(edge_params):
        if verbose:
            print(f"  Using LLM-suggested bounds for {len(llm_bounds)} parameters:")
        for i, (ep, lb) in enumerate(zip(edge_params, llm_bounds)):
            old_lower, old_upper = ep['lower'], ep['upper']
            new_lower, new_upper = lb['lower'], lb['upper']
            # Enforce minimum bound width: at least 20% of value on each side
            val = abs(ep['value']) if ep['value'] != 0 else 1.0
            min_width = val * 0.2
            if new_upper - new_lower < min_width:
                center = (new_lower + new_upper) / 2
                new_lower = center - min_width / 2
                new_upper = center + min_width / 2
            ep['lower'] = new_lower
            ep['upper'] = new_upper
            if verbose:
                print(f"    {ep['name']}: [{old_lower:.4f}, {old_upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}]")
    elif llm_bounds is not None:
        if verbose:
            print(f"  LLM bounds count ({len(llm_bounds)}) doesn't match param count "
                  f"({len(edge_params)}), using default +/-30% bounds")

    # Build Nevergrad parametrization with init values at data-estimated centers
    ng_params = {}
    for i, p in enumerate(edge_params):
        scalar = ng.p.Scalar(lower=p['lower'], upper=p['upper'], init=p['value'])
        ng_params[f"param_{i}"] = scalar

    parametrization = ng.p.Instrumentation(**ng_params)

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(16, cpu_count // 4))

    if budget is None:
        base = max(500, len(edge_params) * 500)
        budget = int(base * math.sqrt(num_workers)) if num_workers > 1 else base
        budget = min(budget, 10000)

    if verbose:
        print(f"  Budget: {budget}, Workers: {num_workers}")

    # Use CMA — better for refining continuous params with good initial estimates
    optimizer = ng.optimizers.CMA(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers,
    )

    objective_fn = functools.partial(
        _nevergrad_edge_objective,
        template_str=template_str,
        npz_file_paths=npz_file_paths,
        dt=dt,
        total_time=total_time,
        verbose=False,  # quiet during optimization
        param_count=len(edge_params),
    )

    if verbose:
        print(f"  Running optimization...")

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            recommendation = optimizer.minimize(objective_fn, executor=executor, batch_mode=False)
    else:
        recommendation = optimizer.minimize(objective_fn)

    # Reconstruct optimized spec
    optimal_array = np.array([recommendation.kwargs[f"param_{i}"] for i in range(len(edge_params))])
    optimal_spec = _reconstruct_from_template(template_str, optimal_array)

    opt_params = {
        edge_params[i]['name']: recommendation.kwargs[f"param_{i}"]
        for i in range(len(edge_params))
    }

    # Evaluate final spec to get actual error (recommendation.loss can be None for CMA)
    final_error = recommendation.loss
    if final_error is None:
        fn = functools.partial(
            _nevergrad_edge_objective,
            template_str=template_str,
            npz_file_paths=npz_file_paths,
            dt=dt,
            total_time=total_time,
            verbose=False,
            param_count=len(edge_params),
        )
        final_error = fn(**recommendation.kwargs)

    if verbose:
        print(f"\n  Optimal edge parameters:")
        for name, val in opt_params.items():
            print(f"    {name}: {val:.6f}")
        print(f"  Final error: {final_error:.6f}")

    return {
        'ha_spec': optimal_spec,
        'error': final_error,
        'edge_params': opt_params,
    }


# ============================================================================
# Combined Pipeline
# ============================================================================

def optimize_ha_sindy_nevergrad(
    ha_spec: Dict[str, Any],
    input_data_path: str,
    train_num: int = 1,
    poly_degree: int = 3,
    sindy_threshold: float = 0.05,
    nevergrad_budget: Optional[int] = None,
    num_workers: Optional[int] = None,
    edge_estimation_model: str = "gemini/gemini-flash-lite-latest",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: SINDy for mode ODEs + Nevergrad for edge parameters.

    Args:
        ha_spec: Initial HA specification
        input_data_path: Path to data directory (ground truth in {path}_g/)
        train_num: Number of ground truth files to use
        poly_degree: SINDy polynomial degree
        sindy_threshold: SINDy sparsity threshold
        nevergrad_budget: Edge optimization budget (auto if None)
        num_workers: Parallel workers for Nevergrad
        edge_estimation_model: LiteLLM model ID for edge estimation LLM calls
        verbose: Print progress

    Returns:
        Dict with 'ha_spec', 'sindy_spec', 'error', 'edge_params'
    """
    if verbose:
        print("=" * 70)
        print("  SINDy + Nevergrad HA Parameter Optimization Pipeline")
        print("=" * 70)

    # Resolve ground truth paths
    gt_path = input_data_path.rstrip('/') + '_g'
    if not os.path.isdir(gt_path):
        raise RuntimeError(f"Ground truth directory not found: {gt_path}")

    gt_files = sorted(
        [f for f in os.listdir(gt_path) if f.startswith('ground_truth') and f.endswith('.npz')],
        key=lambda f: int(re.search(r'(\d+)', f).group(1)) if re.search(r'(\d+)', f) else 0
    )
    npz_paths = [os.path.join(gt_path, f) for f in gt_files[:train_num]]

    if not npz_paths:
        raise RuntimeError(f"No ground truth files found in {gt_path}")

    if verbose:
        print(f"\n  Using {len(npz_paths)} ground truth file(s):")
        for p in npz_paths:
            print(f"    - {p}")

    # --- Phase 1: SINDy Agent ---
    sindy_spec = sindy_fit_modes(
        ha_spec=ha_spec,
        npz_file_paths=npz_paths,
        poly_degree=poly_degree,
        sindy_threshold=sindy_threshold,
        verbose=verbose,
    )

    # Evaluate SINDy-only result
    if verbose:
        print("\n" + "-" * 60)
        print("[Evaluation] SINDy-fitted spec (before edge optimization):")
        for npz_path in npz_paths:
            try:
                ev = HAEvaluator(sindy_spec, npz_path, dt=sindy_spec['config']['dt'],
                                 total_time=sindy_spec['config']['total_time'])
                ev.load_ground_truth()
                ev.simulate()
                m = ev.compute_metrics()
                print(f"  {os.path.basename(npz_path)}: mean_diff={m.get('mean_diff', 'N/A')}")
            except Exception as e:
                print(f"  {os.path.basename(npz_path)}: evaluation failed: {e}")

    # --- Phase 1.5: LLM-based edge parameter estimation ---
    edge_spec, llm_bounds = estimate_edge_params_with_llm(
        ha_spec=sindy_spec,
        npz_file_paths=npz_paths,
        model_id=edge_estimation_model,
        verbose=verbose,
    )

    # Evaluate after edge estimation
    if verbose:
        estimation_method = "LLM" if llm_bounds is not None else "fallback (unchanged)"
        print("\n" + "-" * 60)
        print(f"[Evaluation] After {estimation_method} edge estimation:")
        for npz_path in npz_paths:
            try:
                ev = HAEvaluator(edge_spec, npz_path, dt=edge_spec['config']['dt'],
                                 total_time=edge_spec['config']['total_time'])
                ev.load_ground_truth()
                ev.simulate()
                m = ev.compute_metrics()
                print(f"  {os.path.basename(npz_path)}: mean_diff={m.get('mean_diff', 'N/A')}")
            except Exception as e:
                print(f"  {os.path.basename(npz_path)}: evaluation failed: {e}")

    # --- Phase 2: Nevergrad refinement (fine-tuning around estimates) ---
    result = nevergrad_optimize_edges(
        ha_spec=edge_spec,
        npz_file_paths=npz_paths,
        budget=nevergrad_budget,
        num_workers=num_workers,
        llm_bounds=llm_bounds,
        verbose=verbose,
    )

    result['sindy_spec'] = sindy_spec
    result['edge_estimated_spec'] = edge_spec

    if verbose:
        print("\n" + "=" * 70)
        print("  FINAL OPTIMIZED HA SPECIFICATION")
        print("=" * 70)
        print(json.dumps(result['ha_spec'], indent=2))

    return result


# ============================================================================
# Demo: Duffing oscillator
# ============================================================================

if __name__ == "__main__":
    input_data_path = "data_all/non_linear/duffing"
    gt_dir = input_data_path + '_g'

    if not os.path.isdir(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    # Same initial_ha_spec from optimize_ha_params.py
    initial_ha_spec = {
        "automaton": {
            "var": "x",
            "input": "u",
            "mode": [
                {
                    "id": 1,
                    "eq": "x[2] = u - 0.3 * x[1] + x[0] - 1* x[0] ** 3"
                },
                {
                    "id": 2,
                    "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1* x[0] ** 3"
                }
            ],
            "edge": [
                {
                    "direction": "1 -> 2",
                    "condition": "abs(x) <= 0.9",
                    "reset": {
                        "x": ["", "x[1] * 0.3"]
                    }
                },
                {
                    "direction": "2 -> 1",
                    "condition": "abs(x) >= 1.2",
                    "reset": {
                        "x": ["", "x[1] * 0.95"]
                    }
                }
            ]
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "order": 2,
            "need_reset": True
        }
    }

    print("Initial HA Specification (with WRONG parameters):")
    print(json.dumps(initial_ha_spec, indent=2))
    print()

    result = optimize_ha_sindy_nevergrad(
        ha_spec=initial_ha_spec,
        input_data_path=input_data_path,
        train_num=1,
        poly_degree=3,
        sindy_threshold=0.05,
        edge_estimation_model="gemini/gemini-flash-lite-latest",
        verbose=True,
    )

    print(f"\nFinal error: {result.get('error', 'N/A')}")
