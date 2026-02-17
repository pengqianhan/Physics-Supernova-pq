"""
Combined SINDy + Nevergrad Hybrid Automaton Parameter Optimization

Pipeline:
  1. SINDy Agent: Fits mode ODEs (mode.eq) from trajectory data using PySINDy
  2. Nevergrad Agent: Optimizes edge parameters (edge.condition & edge.reset)
     using gradient-free optimization against ground truth

Usage:
    python optimize_sindy_nevergrad.py  # Runs duffing oscillator demo
"""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/Dainarx_code'))
from HA_evaluation import HAEvaluator


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
# Data-Driven Edge Parameter Estimation
# ============================================================================

def estimate_edge_params_from_data(
    ha_spec: Dict[str, Any],
    npz_file_paths: List[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Estimate edge (guard + reset) parameters directly from change point data.

    At each change point we can observe:
      - The state values where transitions fire (→ guard thresholds)
      - The ratio of derivatives before/after the transition (→ reset coefficients)

    Returns:
        Updated HA spec with data-estimated edge parameters
    """
    if verbose:
        print("\n" + "=" * 60)
        print("[Data-Driven] Estimating edge parameters from change points")
        print("=" * 60)

    automaton = ha_spec['automaton']
    config = ha_spec.get('config', {})
    dt = config.get('dt', 0.001)
    order = config.get('order', 2)

    var_str = automaton.get('var', 'x')
    var_names = [v.strip() for v in var_str.split(',') if v.strip()]
    n_vars = len(var_names)

    # Collect transition data from all NPZ files
    # transition_data[(from_mode, to_mode)] = list of {state_before, deriv_before, state_after, deriv_after}
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
                # State and derivatives just before transition
                entry[f'{vn}_vals_before'] = [all_derivs[v][k][cp_idx - 1] for k in range(order + 1)]
                # State and derivatives just after transition
                entry[f'{vn}_vals_after'] = [all_derivs[v][k][min(cp_idx, len(mode) - 1)] for k in range(order + 1)]

            transition_data.setdefault((m_before, m_after), []).append(entry)

    if verbose:
        for key, entries in sorted(transition_data.items()):
            print(f"  Transition {key[0]}->{key[1]}: {len(entries)} observations")

    # Now update edge parameters based on collected data
    updated_spec = json.loads(json.dumps(ha_spec))
    for edge in updated_spec['automaton'].get('edge', []):
        direction = edge.get('direction', '')
        match = re.match(r'(\d+)\s*->\s*(\d+)', direction)
        if not match:
            continue
        from_mode, to_mode = int(match.group(1)), int(match.group(2))
        key = (from_mode, to_mode)

        if key not in transition_data or not transition_data[key]:
            if verbose:
                print(f"  Edge {from_mode}->{to_mode}: no transition data, keeping original")
            continue

        entries = transition_data[key]

        # --- Estimate guard threshold ---
        if 'condition' in edge:
            cond = edge['condition']
            estimated_cond = _estimate_guard_from_data(cond, entries, var_names, verbose)
            if estimated_cond:
                edge['condition'] = estimated_cond
                if verbose:
                    print(f"  Edge {from_mode}->{to_mode} guard: '{cond}' -> '{estimated_cond}'")

        # --- Estimate reset coefficients ---
        if 'reset' in edge:
            estimated_reset = _estimate_reset_from_data(
                edge['reset'], entries, var_names, order, verbose
            )
            if estimated_reset:
                edge['reset'] = estimated_reset
                if verbose:
                    print(f"  Edge {from_mode}->{to_mode} reset: {estimated_reset}")

    return updated_spec


def _estimate_guard_from_data(
    condition: str,
    entries: List[Dict],
    var_names: List[str],
    verbose: bool,
) -> Optional[str]:
    """
    Estimate the guard threshold from transition observations.

    Supports patterns like:
      - "abs(x) <= 0.9"  → estimate threshold from |x| at transition points
      - "x >= 1.2"       → estimate from x values
      - "x1 <= 5"        → estimate from x1 values
    """
    # Parse the condition to find variable and comparison
    # Pattern: optional abs(), variable name, comparison, number
    m = re.match(
        r'(abs\()?\s*([a-zA-Z_]\w*(?:\[\d+\])?)\s*\)?\s*(<=|>=|<|>|==)\s*([\d.eE+-]+)',
        condition.strip()
    )
    if not m:
        return None

    has_abs = m.group(1) is not None
    var_ref = m.group(2)
    comparator = m.group(3)

    # Find which variable this refers to
    # var_ref could be "x", "x1", "x[0]", etc.
    var_idx = 0
    deriv_idx = 0
    for i, vn in enumerate(var_names):
        if var_ref.startswith(vn):
            var_idx = i
            # Check for derivative index: x[1] means first derivative
            bracket_match = re.search(r'\[(\d+)\]', var_ref)
            if bracket_match:
                deriv_idx = int(bracket_match.group(1))
            break

    # Collect the variable values at each transition
    vn = var_names[var_idx]
    values = []
    for e in entries:
        before_vals = e.get(f'{vn}_vals_before', [])
        if deriv_idx < len(before_vals):
            val = before_vals[deriv_idx]
            values.append(abs(val) if has_abs else val)

    if not values:
        return None

    # Use median as robust estimate
    threshold = float(np.median(values))

    if verbose:
        print(f"    Guard '{condition}': observed values {[f'{v:.4f}' for v in values[:5]]}... "
              f"-> median={threshold:.6f}")

    abs_prefix = "abs(" if has_abs else ""
    abs_suffix = ")" if has_abs else ""
    return f"{abs_prefix}{var_ref}{abs_suffix} {comparator} {threshold:.6f}"


def _estimate_reset_from_data(
    reset: Dict[str, Any],
    entries: List[Dict],
    var_names: List[str],
    order: int,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """
    Estimate reset coefficients from before/after derivative ratios.

    For a reset like "x[1] * coef", computes coef = deriv_after / deriv_before.
    """
    estimated = {}
    for var_name, reset_val in reset.items():
        if not isinstance(reset_val, list):
            estimated[var_name] = reset_val
            continue

        new_list = []
        for k, item in enumerate(reset_val):
            if not isinstance(item, str) or not item.strip():
                new_list.append(item)
                continue

            # Try to detect pattern: "var[k] * coef" or "coef * var[k]"
            # and estimate coef from data
            coef_match = re.search(
                r'([a-zA-Z_]\w*)\[(\d+)\]\s*\*\s*([\d.eE+-]+)', item
            )
            if not coef_match:
                coef_match_rev = re.search(
                    r'([\d.eE+-]+)\s*\*\s*([a-zA-Z_]\w*)\[(\d+)\]', item
                )
                if coef_match_rev:
                    ref_var = coef_match_rev.group(2)
                    deriv_k = int(coef_match_rev.group(3))
                else:
                    new_list.append(item)
                    continue
            else:
                ref_var = coef_match.group(1)
                deriv_k = int(coef_match.group(2))

            # Find the matching variable index
            v_idx = None
            for i, vn in enumerate(var_names):
                if ref_var == vn:
                    v_idx = i
                    break
            if v_idx is None:
                new_list.append(item)
                continue

            # Compute ratio of after/before for this derivative
            vn = var_names[v_idx]
            ratios = []
            for e in entries:
                before = e.get(f'{vn}_vals_before', [])
                after = e.get(f'{vn}_vals_after', [])
                if deriv_k < len(before) and deriv_k < len(after):
                    b_val = before[deriv_k]
                    a_val = after[deriv_k]
                    if abs(b_val) > 1e-8:
                        ratios.append(a_val / b_val)

            if ratios:
                coef = float(np.median(ratios))
                if verbose:
                    print(f"    Reset {var_name}[{k}]: ratios={[f'{r:.4f}' for r in ratios[:5]]}... "
                          f"-> median coef={coef:.6f}")
                # Check for negative sign prefix in original
                neg_prefix = "-" if item.strip().startswith("-") else ""
                new_list.append(f"{neg_prefix}{ref_var}[{deriv_k}] * {abs(coef):.6f}")
            else:
                new_list.append(item)

        estimated[var_name] = new_list

    return estimated


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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Nevergrad Agent: Optimize edge condition and reset parameters.

    Args:
        ha_spec: HA spec (with SINDy-fitted mode equations)
        npz_file_paths: Ground truth NPZ file paths
        budget: Optimization budget (auto-calculated if None)
        num_workers: Parallel workers (auto if None)
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

    # --- Phase 1.5: Data-driven edge estimation ---
    data_spec = estimate_edge_params_from_data(
        ha_spec=sindy_spec,
        npz_file_paths=npz_paths,
        verbose=verbose,
    )

    # Evaluate after data-driven estimation
    if verbose:
        print("\n" + "-" * 60)
        print("[Evaluation] After data-driven edge estimation:")
        for npz_path in npz_paths:
            try:
                ev = HAEvaluator(data_spec, npz_path, dt=data_spec['config']['dt'],
                                 total_time=data_spec['config']['total_time'])
                ev.load_ground_truth()
                ev.simulate()
                m = ev.compute_metrics()
                print(f"  {os.path.basename(npz_path)}: mean_diff={m.get('mean_diff', 'N/A')}")
            except Exception as e:
                print(f"  {os.path.basename(npz_path)}: evaluation failed: {e}")

    # --- Phase 2: Nevergrad refinement (fine-tuning around data estimates) ---
    result = nevergrad_optimize_edges(
        ha_spec=data_spec,
        npz_file_paths=npz_paths,
        budget=nevergrad_budget,
        num_workers=num_workers,
        verbose=verbose,
    )

    result['sindy_spec'] = sindy_spec
    result['data_estimated_spec'] = data_spec

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
        verbose=True,
    )

    print(f"\nFinal error: {result.get('error', 'N/A')}")
