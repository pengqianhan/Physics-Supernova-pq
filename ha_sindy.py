"""
ha_sindy.py — Simulate a Hybrid Automaton using PySINDy + SINDy identification.

Pipeline:
  1. Build PySINDy models per mode with known ODE coefficients
  2. Simulate using model.predict() + RK4, with manual mode switching
  3. Fit PySINDy per mode to recover coefficients from trajectory data
  4. Compare identified vs ground truth equations

Usage:
    python ha_sindy.py
"""

import numpy as np
import pysindy as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import cos


# ============================================================================
# HA Specification
# ============================================================================

HA_SPEC = {
    "automaton": {
        "var": "x",
        "input": "u",
        "mode": [
            {"id": 1, "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3"},
            {"id": 2, "eq": "x[2] = u - 0.2 * x[1] + x[0] - 0.5 * x[0] ** 3"},
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "abs(x) <= 0.8",
                "reset": {"x": ["", "x[1] * 0.95"]},
            },
            {
                "direction": "2 -> 1",
                "condition": "abs(x) >= 1.2",
                "reset": {"x": ["", "x[1] * 0.95"]},
            },
        ],
    },
    "config": {"dt": 0.001, "total_time": 10.0, "order": 2},
}

INIT_STATES = [
    {"mode": 1, "x": [4],   "u": "0.5 * cos(1.2 * t)"},
    {"mode": 1, "x": [7.8], "u": "0.65 * cos(1.0 * t)"},
    {"mode": 1, "x": [7.4], "u": "0.7 * cos(1.3 * t)"},
]


# ============================================================================
# Step 1: Build PySINDy models with known coefficients
# ============================================================================

def build_mode_models():
    """Build a PySINDy SINDy model per mode with ground-truth coefficients.

    State y = [x, v]  (position, velocity).
    Autonomous part (without external input u):
      Mode 1: dy/dt = [v,  x - 0.5*v - 1.5*x^3]
      Mode 2: dy/dt = [v,  x - 0.2*v - 0.5*x^3]
    The forcing term u(t) is added externally during simulation.
    """
    mode_params = {
        1: (-0.5, -1.5),   # (damping on v, cubic coefficient)
        2: (-0.2, -0.5),
    }
    models = {}

    for mid, (damping, cubic) in mode_params.items():
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=3, include_bias=True),
            optimizer=ps.STLSQ(threshold=0.01),
        )
        # Fit on dummy data to initialise library internals
        n = 50
        model.fit(np.random.randn(n, 2), x_dot=np.random.randn(n, 2),
                  t=0.001, feature_names=["x", "v"])

        # Overwrite coefficients to match known ODEs
        fnames = model.get_feature_names()
        coef = np.zeros((2, len(fnames)))
        for i, fn in enumerate(fnames):
            fn = fn.strip()
            if fn == "v":
                coef[0, i] = 1.0        # dx/dt = v
                coef[1, i] = damping     # dv/dt: damping * v
            elif fn == "x":
                coef[1, i] = 1.0        # dv/dt: + x
            elif fn == "x^3":
                coef[1, i] = cubic       # dv/dt: cubic * x^3
        model.optimizer.coef_ = coef
        models[mid] = model

        if mid == 1:
            print(f"  Library features: {fnames}")
        print(f"  Mode {mid}: damping={damping}, cubic={cubic}")

    return models


# ============================================================================
# Step 2: Simulate using PySINDy models (RK4 + mode switching)
# ============================================================================

def _sindy_rk4_step(model, state, u_val, dt):
    """One RK4 step: dy/dt = model.predict(y) + [0, u]."""
    forcing = np.array([0.0, u_val])

    def rhs(y):
        return np.asarray(model.predict(y.reshape(1, -1))).flatten() + forcing

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)


def simulate(mode_models, ha_spec, init_states):
    """Simulate HA using PySINDy model.predict() in an RK4 loop."""
    dt = ha_spec["config"]["dt"]
    n_steps = int(ha_spec["config"]["total_time"] / dt)
    trajectories = []

    for idx, init in enumerate(init_states):
        u_func = eval(f"lambda t: {init['u']}")
        state = np.array([float(init["x"][0]), 0.0])  # [x, v=0]
        current_mode = init["mode"]

        x_arr = np.zeros(n_steps)
        u_arr = np.zeros(n_steps)
        mode_arr = np.zeros(n_steps, dtype=int)
        change_points = [0]

        for step in range(n_steps):
            t = (step + 1) * dt
            u_val = u_func(t)

            # Integrate one step using PySINDy model
            state = _sindy_rk4_step(
                mode_models[current_mode], state, u_val, dt
            )

            x_arr[step] = state[0]
            u_arr[step] = u_val
            mode_arr[step] = current_mode

            # Guard checks + mode switching
            switched = False
            if current_mode == 1 and abs(state[0]) <= 0.8:
                state[1] *= 0.95        # reset velocity
                current_mode = 2
                switched = True
            elif current_mode == 2 and abs(state[0]) >= 1.2:
                state[1] *= 0.95
                current_mode = 1
                switched = True

            if switched:
                change_points.append(step + 1)

        change_points.append(n_steps)
        trajectories.append({
            "x": x_arr, "u": u_arr,
            "mode": mode_arr, "cp": np.array(change_points),
        })
        print(f"  Sample {idx:2d}: x0={init['x'][0]:5.1f}, "
              f"{len(change_points) - 2} mode switches")

    return trajectories


# ============================================================================
# Step 2: Segment by mode + fit SINDy
# ============================================================================

def fit_sindy(trajectories, dt, poly_degree=3, threshold=0.05,
              min_seg=80, trim=10):
    """Segment trajectories by mode, fit one SINDy model per mode."""
    segments = {}  # mode_id -> list of (X_features, y_target)

    for traj in trajectories:
        x, u, mode, cp = traj["x"], traj["u"], traj["mode"], traj["cp"]

        for j in range(len(cp) - 1):
            s, e = int(cp[j]), int(cp[j + 1])
            if e - s < min_seg + 2 * trim:
                continue
            mid = int(mode[s])

            # Compute derivatives within this segment only
            # (avoids contamination from mode-switch velocity resets)
            seg_x = x[s:e]
            seg_x_dot = np.gradient(seg_x, dt)
            seg_x_ddot = np.gradient(seg_x_dot, dt)
            seg_u = u[s:e]

            # Trim edges to discard boundary artifacts
            X = np.column_stack([seg_x, seg_x_dot, seg_u])[trim:-trim]
            y = seg_x_ddot[trim:-trim, np.newaxis]
            segments.setdefault(mid, []).append((X, y))

    feature_names = ["x[0]", "x[1]", "u"]
    models = {}

    for mid in sorted(segments):
        segs = segments[mid]
        X_all = np.vstack([s[0] for s in segs])
        y_all = np.vstack([s[1] for s in segs])
        print(f"\n  Mode {mid}: {len(segs)} segments, {X_all.shape[0]} total points")

        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=poly_degree),
            optimizer=ps.STLSQ(threshold=threshold),
        )
        model.fit(X_all, x_dot=y_all, t=dt, feature_names=feature_names)
        print("  Identified equation:")
        model.print()
        models[mid] = model

    return models


# ============================================================================
# Step 3: Plot trajectories
# ============================================================================

def plot_trajectories(trajectories, dt, save_path="ha_sindy_plot.png", n_show=4):
    """Plot sample trajectories colored by mode."""
    n = min(n_show, len(trajectories))
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = {1: "tab:blue", 2: "tab:orange"}

    for i, ax in enumerate(axes):
        traj = trajectories[i]
        t = np.arange(len(traj["x"])) * dt
        for mid in [1, 2]:
            mask = traj["mode"] == mid
            if mask.any():
                ax.scatter(t[mask], traj["x"][mask], c=colors[mid],
                           s=0.3, label=f"Mode {mid}")
        ax.plot(t, traj["u"], "k--", lw=0.5, alpha=0.4, label="u (input)")
        ax.set_ylabel(f"x  (sample {i})")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, markerscale=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Hybrid Automaton Trajectories (colored by mode)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# Step 4: Compare SINDy results with ground truth
# ============================================================================

def print_comparison(models):
    """Print side-by-side comparison of ground truth and SINDy-identified equations."""
    ground_truth = {
        1: {"u": 1.0, "x[1]": -0.5, "x[0]": 1.0, "x[0]^3": -1.5},
        2: {"u": 1.0, "x[1]": -0.2, "x[0]": 1.0, "x[0]^3": -0.5},
    }

    print("\n" + "=" * 65)
    print("  Ground Truth vs SINDy Identified Equations")
    print("=" * 65)

    for mid in sorted(ground_truth):
        gt = ground_truth[mid]
        gt_str = " ".join(f"{v:+.1f}*{k}" for k, v in gt.items())
        print(f"\n  Mode {mid} (GT):    x'' = {gt_str}")

        if mid in models:
            coefs = models[mid].coefficients()[0]
            names = models[mid].get_feature_names()
            terms = [f"{c:+.4f}*{n}" for c, n in zip(coefs, names) if abs(c) > 1e-6]
            sindy_str = " ".join(terms) if terms else "0"
            print(f"  Mode {mid} (SINDy): x'' = {sindy_str}")

            # Per-coefficient error analysis
            print(f"  Coefficient errors:")
            for gt_name, gt_val in gt.items():
                matched = False
                for c, n in zip(coefs, names):
                    if gt_name == n:
                        err = abs(c - gt_val)
                        print(f"    {gt_name:>8s}: GT={gt_val:+.4f}, SINDy={c:+.4f}, |err|={err:.6f}")
                        matched = True
                        break
                if not matched:
                    # Match x[0]^3 as "x[0] x[0] x[0]" (SINDy product notation)
                    for c, n in zip(coefs, names):
                        if "^3" in gt_name and n.count(gt_name.split("^")[0]) == 3:
                            err = abs(c - gt_val)
                            print(f"    {gt_name:>8s}: GT={gt_val:+.4f}, SINDy={c:+.4f}, |err|={err:.6f}")
                            matched = True
                            break
                if not matched:
                    print(f"    {gt_name:>8s}: GT={gt_val:+.4f}, SINDy=NOT FOUND")
        else:
            print(f"  Mode {mid} (SINDy): no data / not fitted")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Hybrid Automaton Simulation (PySINDy) + SINDy Identification")
    print("=" * 65)

    # 1. Build PySINDy models with known coefficients
    print("\n[1/4] Building PySINDy mode models ...")
    mode_models = build_mode_models()

    # 2. Simulate using PySINDy models
    print("\n[2/4] Simulating hybrid automaton (15 initial conditions) ...")
    trajs = simulate(mode_models, HA_SPEC, INIT_STATES)

    # 3. Plot
    print("\n[3/4] Plotting sample trajectories ...")
    plot_trajectories(trajs, HA_SPEC["config"]["dt"])

    # 4. SINDy identification from trajectory data
    print("\n[4/4] Fitting SINDy per mode ...")
    models = fit_sindy(trajs, HA_SPEC["config"]["dt"])

    # 5. Compare
    print_comparison(models)
