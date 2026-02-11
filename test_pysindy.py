"""
PySINDy-based Hybrid Automaton System Identification

Discovers governing equations for each mode of a Hybrid Automaton
using Sparse Identification of Nonlinear Dynamics (SINDy).

Replaces the Nevergrad black-box optimization in optimize_ha_params.py
with direct data-driven equation discovery:
  1. Load ground truth trajectories (.npz)
  2. Compute numerical derivatives (position -> velocity -> acceleration)
  3. Segment data by mode (using mode labels + boundary trimming)
  4. Fit a separate SINDy model per mode (STLSQ sparse regression)
  5. Estimate guard conditions & reset maps from transition points
  6. Construct complete HA JSON specification
  7. Evaluate against ground truth using HAEvaluator

Usage:
    python test_pysindy.py
"""

import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter
from collections import defaultdict
import sys
import os
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/Dainarx_code'))
from HA_evaluation import HAEvaluator


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_trajectory(npz_path):
    """Load raw trajectory data from NPZ file."""
    data = np.load(npz_path)
    x = data['state'][0]
    inp = data['input']
    u = inp[0] if inp.ndim == 2 else inp
    mode = data['mode']
    cps = data['change_points']
    return x, u, mode, cps


def segment_and_differentiate(x, u, mode, dt, margin=30, savgol_window=21):
    """Segment by mode FIRST, then compute derivatives within each segment.

    This is critical: velocity resets at mode transitions cause discontinuities.
    Computing derivatives across these boundaries corrupts the data.
    By differentiating within each contiguous mode segment, we get clean
    velocity and acceleration estimates.

    Args:
        x: position array (n_steps,)
        u: input array (n_steps,)
        mode: mode labels (n_steps,)
        dt: time step
        margin: points to trim from each segment boundary
        savgol_window: Savitzky-Golay filter window (odd integer)

    Returns dict: mode_id -> list of (X, X_dot, U) tuples
    """
    segments = defaultdict(list)
    n = len(mode)
    min_seg_len = savgol_window + 2 * margin + 10

    i = 0
    while i < n:
        m = int(mode[i])
        j = i
        while j < n and mode[j] == m:
            j += 1

        seg_len = j - i
        if seg_len >= min_seg_len:
            # Extract raw segment data
            x_seg = x[i:j]
            u_seg = u[i:j]

            # Compute derivatives WITHIN this segment only
            wl = min(savgol_window, seg_len // 2 * 2 - 1)
            if wl < 5:
                wl = 5
            v_seg = savgol_filter(x_seg, window_length=wl, polyorder=5,
                                  deriv=1, delta=dt)
            a_seg = savgol_filter(x_seg, window_length=wl, polyorder=5,
                                  deriv=2, delta=dt)

            # Trim boundary artifacts
            s = margin
            e = seg_len - margin
            if e - s >= 20:
                X = np.column_stack([x_seg[s:e], v_seg[s:e]])
                X_dot = np.column_stack([v_seg[s:e], a_seg[s:e]])
                U = u_seg[s:e].reshape(-1, 1)
                segments[m].append((X, X_dot, U))

        i = j
    return dict(segments)


# =============================================================================
# SINDy Fitting
# =============================================================================

def fit_sindy_per_mode(mode_segments, dt, poly_degree=3, threshold=0.05):
    """Fit a separate SINDy model for each mode using STLSQ.

    STLSQ (Sequential Thresholded Least Squares) performs sparse regression:
    it iteratively removes terms with coefficients below the threshold,
    promoting parsimony in the discovered equations.
    """
    models = {}

    for mode_id in sorted(mode_segments.keys()):
        segs = mode_segments[mode_id]
        total_pts = sum(s[0].shape[0] for s in segs)
        print(f"\n  Mode {mode_id}: {len(segs)} segments, {total_pts} data points")

        # Split each segment tuple (x, x_dot, u) into separate lists for SINDy.fit.
        x_list = [s[0] for s in segs]   # State trajectories per segment
        xd_list = [s[1] for s in segs]  # State-derivative trajectories per segment
        u_list = [s[2] for s in segs]   # Control-input trajectories per segment

        library = ps.PolynomialLibrary(degree=poly_degree, include_bias=False)
        optimizer = ps.STLSQ(threshold=threshold, alpha=0.01)

        model = ps.SINDy(
            feature_library=library,
            optimizer=optimizer,
        )

        # feature_names must cover state + control dimensions
        n_control = u_list[0].shape[1]
        all_names = ['x', 'v'] + [f'u{i}' for i in range(n_control)]

        if len(x_list) == 1:
            model.fit(x_list[0], t=dt, x_dot=xd_list[0], u=u_list[0],
                      feature_names=all_names)
        else:
            model.fit(x_list, t=dt, x_dot=xd_list, u=u_list,
                      feature_names=all_names)

        model.print()

        # Debug info
        fnames = model.get_feature_names()
        coefs = model.coefficients()
        print(f"    Feature names ({len(fnames)}): {fnames}")
        print(f"    Coefs shape: {coefs.shape}")
        if coefs.shape[1] > len(fnames):
            n_extra = coefs.shape[1] - len(fnames)
            print(f"    ({n_extra} extra columns = control input features)")

        models[mode_id] = model

    return models


# =============================================================================
# SINDy -> HA Equation Conversion
# =============================================================================

def _map_sindy_var(name, var='x', inp='u'):
    """Map a single SINDy variable token to HA notation."""
    if name == 'x':
        return f'{var}[0]'
    if name == 'v':
        return f'{var}[1]'
    if name.startswith('u'):
        return inp
    if name == '1':
        return '1'
    return name


def sindy_feature_to_ha(fname, var='x', inp='u'):
    """Convert PySINDy feature name to HA equation notation.

    PySINDy uses space-separated tokens for products and ^ for powers:
        'x'     -> 'x[0]'
        'v'     -> 'x[1]'
        'u0'    -> 'u'
        'x^3'   -> 'x[0] ** 3'
        'x v'   -> 'x[0] * x[1]'
        'x^2 v' -> 'x[0] ** 2 * x[1]'
    """
    tokens = fname.strip().split(' ')
    parts = []
    for tok in tokens:
        if '^' in tok:
            base, power = tok.split('^')
            parts.append(f'{_map_sindy_var(base, var, inp)} ** {power}')
        else:
            parts.append(_map_sindy_var(tok, var, inp))
    return ' * '.join(parts)


def build_ode_from_sindy(model, eq_idx, var='x', inp='u'):
    """Extract ODE right-hand side string from a fitted SINDy model.

    For a 2nd-order system with state [x, v]:
      eq_idx=0: dx/dt equation (should be ~v, sanity check)
      eq_idx=1: dv/dt equation (the actual ODE we want)
    """
    coefs = model.coefficients()[eq_idx]
    fnames = list(model.get_feature_names())

    # PySINDy may store control input coefficients beyond the library features
    n_extra = len(coefs) - len(fnames)
    if n_extra > 0:
        for i in range(n_extra):
            fnames.append(f'u{i}')

    # Collect non-zero terms
    terms = []
    for c, fn in zip(coefs, fnames):
        if abs(c) < 1e-6:
            continue
        ha_feat = sindy_feature_to_ha(fn, var, inp)
        terms.append((c, ha_feat))

    if not terms:
        return '0'

    # Build readable equation string
    parts = []
    for i, (c, feat) in enumerate(terms):
        is_unit = abs(abs(c) - 1.0) < 0.05
        if is_unit:
            if c > 0:
                parts.append(f'+ {feat}' if i > 0 else feat)
            else:
                parts.append(f'- {feat}')
        else:
            if c > 0:
                parts.append(f'+ {c:.4f} * {feat}' if i > 0 else f'{c:.4f} * {feat}')
            else:
                parts.append(f'- {abs(c):.4f} * {feat}')

    return ' '.join(parts)


# =============================================================================
# Guard & Reset Estimation
# =============================================================================

def collect_transition_data(x, v, mode, cps):
    """Collect state values at mode transitions for guard estimation.

    Uses a small offset (5 steps) from the transition point to get
    clean velocity estimates (avoiding reset discontinuity artifacts).
    Also tracks |x| before the transition for guard direction heuristic.
    """
    transitions = defaultdict(
        lambda: {'x': [], 'v_before': [], 'v_after': [], 'abs_x_before': []}
    )

    for cp in cps:
        if cp <= 20 or cp >= len(mode) - 5:
            continue
        from_m = int(mode[cp - 1])
        to_m = int(mode[cp])
        if from_m == to_m:
            continue

        key = (from_m, to_m)
        transitions[key]['x'].append(x[cp])
        transitions[key]['v_before'].append(v[cp - 5])
        transitions[key]['v_after'].append(v[cp + 5])
        # |x| well before transition to determine approach direction
        transitions[key]['abs_x_before'].append(abs(x[cp - 20]))

    return transitions


def estimate_edges(all_transitions, var='x'):
    """Estimate guard conditions and reset maps from transition data.

    Guard heuristic:
      - Threshold: average |x| at transition points
      - Direction (<=  vs >=): if |x| was LARGER before the transition
        than at the transition, the guard is "approaching" (<=).
        Otherwise it's "departing" (>=).
    Reset heuristic: median velocity ratio (v_after / v_before).
    """
    edges = {}

    for (from_m, to_m), data in sorted(all_transitions.items()):
        x_vals = np.array(data['x'])
        v_before = np.array(data['v_before'])
        v_after = np.array(data['v_after'])
        abs_x_before = np.array(data['abs_x_before'])

        avg_abs_x = float(np.mean(np.abs(x_vals)))

        # Determine guard direction: <= or >=
        # If |x| was larger before transition than at transition → approaching → <=
        # If |x| was smaller before transition → departing → >=
        avg_abs_before = float(np.mean(abs_x_before))
        guard_op = '<=' if avg_abs_before > avg_abs_x else '>='

        # Estimate velocity reset ratio
        valid = np.abs(v_before) > 0.01
        if valid.any():
            reset_ratio = float(np.median(v_after[valid] / v_before[valid]))
        else:
            reset_ratio = 1.0

        print(f"  {from_m} -> {to_m}: avg|x| = {avg_abs_x:.4f} "
              f"(before: {avg_abs_before:.4f} -> {guard_op}), "
              f"v_ratio = {reset_ratio:.4f}, n = {len(x_vals)}")

        edges[(from_m, to_m)] = {
            'threshold': avg_abs_x,
            'guard_op': guard_op,
            'reset_ratio': reset_ratio,
        }

    return edges


# =============================================================================
# HA Specification Construction
# =============================================================================

def build_ha_spec(models, edges, config, var='x', inp='u', ode_order=2):
    """Construct complete HA JSON specification from SINDy results."""
    spec = {
        'automaton': {
            'var': var,
            'input': inp,
            'mode': [],
            'edge': []
        },
        'config': config
    }

    # Mode equations: extract the acceleration equation (eq_idx = ode_order - 1)
    for mode_id, model in sorted(models.items()):
        rhs = build_ode_from_sindy(model, eq_idx=ode_order - 1, var=var, inp=inp)
        eq = f'{var}[{ode_order}] = {rhs}'
        spec['automaton']['mode'].append({'id': int(mode_id), 'eq': eq})

    # Edge conditions & resets
    for (from_m, to_m), info in sorted(edges.items()):
        thr = info['threshold']
        op = info.get('guard_op', '<=')
        condition = f'abs({var}) {op} {thr:.4f}'

        edge = {
            'direction': f'{from_m} -> {to_m}',
            'condition': condition,
        }

        rr = info['reset_ratio']
        if abs(rr - 1.0) > 0.05:
            edge['reset'] = {var: ['', f'{var}[1] * {abs(rr):.4f}']}

        spec['automaton']['edge'].append(edge)

    return spec


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_ha_spec(ha_spec, npz_paths, dt, total_time):
    """Evaluate HA spec against ground truth trajectories using HAEvaluator."""
    results = []
    for p in npz_paths:
        try:
            ev = HAEvaluator(
                ha_dict=ha_spec, npz_file_path=p,
                dt=dt, total_time=total_time
            )
            ev.load_ground_truth()
            ev.simulate()
            m = ev.compute_metrics()
            md = m.get('mean_diff', float('inf'))
            print(f"    {os.path.basename(p)}: mean_diff = {md:.6f}")
            results.append(md)
        except Exception as e:
            print(f"    {os.path.basename(p)}: ERROR - {e}")
            results.append(float('inf'))
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    # === Configuration (matches optimize_ha_params.py demo) ===
    input_data_path = "data_all/non_linear/duffing"
    gt_dir = input_data_path + '_g'
    dt = 0.001
    total_time = 10.0
    ode_order = 2
    var = 'x'
    inp = 'u'
    train_num = 5
    eval_num = 1

    ha_config = {
        'dt': dt,
        'total_time': total_time,
        'order': ode_order,
        'need_reset': True
    }

    # Find ground truth files
    if not os.path.isdir(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    gt_files = sorted(
        [f for f in os.listdir(gt_dir)
         if f.startswith('ground_truth') and f.endswith('.npz')],
        key=lambda f: int(re.search(r'(\d+)', f).group(1))
    )

    if not gt_files:
        print(f"Error: No ground_truth_*.npz files in {gt_dir}")
        sys.exit(1)

    print("=" * 70)
    print("PySINDy-based Hybrid Automaton System Identification")
    print("=" * 70)
    print()
    print("Ground truth (Duffing oscillator, from duffing.json):")
    print("  Mode 1: x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3")
    print("  Mode 2: x[2] = u - 0.2 * x[1] + x[0] - 0.5 * x[0] ** 3")
    print("  Edge 1->2: abs(x) <= 0.8, reset x[1] *= 0.95")
    print("  Edge 2->1: abs(x) >= 1.2, reset x[1] *= 0.95")
    print()
    print(f"Data: {gt_dir} ({len(gt_files)} files)")
    print(f"Training: {train_num} files, Evaluation: {eval_num} files")

    # =================================================================
    # Step 1: Load & Segment
    # =================================================================
    print("\n" + "=" * 70)
    print("[Step 1] Loading and segmenting trajectory data")
    print("=" * 70)

    all_segments = defaultdict(list)
    all_transitions = defaultdict(
        lambda: {'x': [], 'v_before': [], 'v_after': [], 'abs_x_before': []}
    )

    for fname in gt_files[:train_num]:
        fpath = os.path.join(gt_dir, fname)
        x, u_arr, mode, cps = load_trajectory(fpath)
        n_trans = max(0, len(cps) - 2)
        print(f"\n  {fname}: {len(x)} steps, modes={np.unique(mode).tolist()}, "
              f"transitions={n_trans}")

        # Segment first, then differentiate within each segment
        segs = segment_and_differentiate(x, u_arr, mode, dt, margin=30)
        for m, seg_list in segs.items():
            all_segments[m].extend(seg_list)

        # For guard estimation, compute velocity over full trajectory
        # (only used at transition points, not for SINDy fitting)
        v_full = np.gradient(x, dt)
        trans = collect_transition_data(x, v_full, mode, cps)
        for key, data in trans.items():
            for field in data:
                all_transitions[key][field].extend(data[field])

    # =================================================================
    # Step 2: Fit SINDy
    # =================================================================
    print("\n" + "=" * 70)
    print("[Step 2] Fitting SINDy models per mode")
    print("=" * 70)

    models = fit_sindy_per_mode(
        all_segments, dt, poly_degree=3, threshold=0.05
    )

    # =================================================================
    # Step 3: Estimate Guards & Resets
    # =================================================================
    print("\n" + "=" * 70)
    print("[Step 3] Estimating guard conditions & resets")
    print("=" * 70)

    edges = estimate_edges(dict(all_transitions), var)

    # =================================================================
    # Step 4: Build HA Spec
    # =================================================================
    print("\n" + "=" * 70)
    print("[Step 4] Constructed HA specification")
    print("=" * 70)

    ha_spec = build_ha_spec(models, edges, ha_config, var, inp, ode_order)
    print(json.dumps(ha_spec, indent=2))

    # =================================================================
    # Step 5: Evaluate
    # =================================================================
    print("\n" + "=" * 70)
    print("[Step 5] Evaluation against ground truth")
    print("=" * 70)

    eval_paths = [os.path.join(gt_dir, f) for f in gt_files[:eval_num]]
    errors = evaluate_ha_spec(ha_spec, eval_paths, dt, total_time)

    valid_errors = [e for e in errors if e != float('inf')]
    if valid_errors:
        avg_err = np.mean(valid_errors)
        print(f"\n  Average mean_diff: {avg_err:.6f}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("Comparison: SINDy-Discovered vs Ground Truth")
    print("=" * 70)

    print("\nGround truth equations (from duffing.json):")
    print("  Mode 1: x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3")
    print("  Mode 2: x[2] = u - 0.2 * x[1] + x[0] - 0.5 * x[0] ** 3")

    print("\nSINDy-discovered equations:")
    for m in ha_spec['automaton']['mode']:
        print(f"  Mode {m['id']}: {m['eq']}")

    print("\nGuard conditions:")
    print("  Ground truth: 1->2: abs(x) <= 0.8,  2->1: abs(x) >= 1.2")
    print("  Discovered:")
    for e in ha_spec['automaton']['edge']:
        reset_str = ""
        if 'reset' in e:
            reset_str = f", reset: {e['reset']}"
        print(f"    {e['direction']}: {e['condition']}{reset_str}")

    return ha_spec


if __name__ == '__main__':
    main()
