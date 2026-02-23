"""
Training data derivative analysis for hybrid automaton datasets.
Plots each state variable's position, velocity, acceleration (single color,
no mode labels), plus input signals, to visually inspect where derivative
discontinuities occur — potential mode transition points.

Usage:
    python data_plot_analysis_train.py                                # default: duffing, sample 0
    python data_plot_analysis_train.py --dataset non_linear/duffing --sample 0
    python data_plot_analysis_train.py --dataset ATVA/ball --sample 3
    python data_plot_analysis_train.py --all                          # ALL datasets, sample 0
    python data_plot_analysis_train.py --all --sample 5
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "data_all"
AUTOMATA_ROOT = "utils/Dainarx_code/automata"
OUTPUT_ROOT = "analysis_plots_train"

SIGNAL_COLORS = ["tab:blue", "tab:green", "tab:red", "tab:orange",
                 "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:cyan"]


def load_config(dataset):
    """Load dt, total_time, var names from the automaton JSON config."""
    json_path = os.path.join(AUTOMATA_ROOT, f"{dataset}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        cfg = json.load(f)
    config = cfg.get("config", {})
    var_str = cfg.get("automaton", {}).get("var", "")
    var_names = [v.strip() for v in var_str.split(",") if v.strip()]
    return {
        "dt": float(config.get("dt", 0.01)),
        "total_time": float(config.get("total_time", 10.0)),
        "var_names": var_names,
    }


def plot_signal(ax, t, y, ylabel, color):
    """Plot a single signal with one color."""
    ax.plot(t, y, color=color, linewidth=0.8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)


def analyze_dataset(dataset, sample_idx, output_root):
    """Generate the derivative analysis plot for one training sample."""
    data_dir = os.path.join(DATA_ROOT, dataset)
    npz_path = os.path.join(data_dir, f"sample_{sample_idx}.npz")
    if not os.path.exists(npz_path):
        print(f"  [SKIP] {npz_path} not found")
        return

    data = np.load(npz_path)
    state = data["state"]       # (num_vars, num_steps)
    inp = data["input"]         # (num_inputs, num_steps) or (num_steps,)

    num_vars = state.shape[0]
    num_steps = state.shape[1]

    # Ensure input is 2D
    if inp.ndim == 1:
        inp = inp.reshape(1, -1)
    num_inputs = inp.shape[0]
    has_input = num_inputs > 0 and inp.size > 0

    # Load config for dt and var names
    cfg = load_config(dataset)
    if cfg:
        dt = cfg["dt"]
        var_names = cfg["var_names"]
    else:
        dt = 0.01
        var_names = [f"x{i+1}" for i in range(num_vars)]

    while len(var_names) < num_vars:
        var_names.append(f"x{len(var_names)+1}")

    t = np.arange(num_steps) * dt

    # Layout: for each state var -> 3 rows (pos, vel, acc), plus input rows
    input_rows = num_inputs if has_input else 0
    total_rows = num_vars * 3 + input_rows
    fig_height = max(8, total_rows * 2.8)
    fig, axes = plt.subplots(total_rows, 1, figsize=(14, fig_height), sharex=True)
    if total_rows == 1:
        axes = [axes]

    system_name = dataset.split("/")[-1]
    fig.suptitle(f"{system_name} — Training Data (sample {sample_idx})",
                 fontsize=14, fontweight="bold", y=0.998)

    row = 0
    for vi in range(num_vars):
        x = state[vi]
        x_dot = np.gradient(x, dt)
        x_ddot = np.gradient(x_dot, dt)
        vname = var_names[vi]
        color = SIGNAL_COLORS[vi % len(SIGNAL_COLORS)]

        plot_signal(axes[row], t, x,      f"{vname}\n(position)", color)
        row += 1
        plot_signal(axes[row], t, x_dot,  f"d{vname}/dt\n(velocity)", color)
        row += 1
        plot_signal(axes[row], t, x_ddot, f"d\u00b2{vname}/dt\u00b2\n(accel.)", color)
        row += 1

    for ui in range(input_rows):
        u = inp[ui]
        plot_signal(axes[row], t, u, f"u{ui+1}\n(input)", "tab:olive")
        row += 1

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    # Save with matching directory structure
    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_{sample_idx}_analysis.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def discover_datasets():
    """Find all training data directories (those without _g suffix that contain sample_*.npz)."""
    datasets = []
    for npz in sorted(glob.glob(os.path.join(DATA_ROOT, "**", "sample_0.npz"), recursive=True)):
        data_dir = os.path.dirname(npz)
        rel = os.path.relpath(data_dir, DATA_ROOT)  # e.g. "non_linear/duffing"
        if not rel.endswith("_g"):
            datasets.append(rel)
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Training data derivative analysis")
    parser.add_argument("--dataset", type=str, default="non_linear/duffing",
                        help="Dataset path relative to data_all, e.g. 'non_linear/duffing', 'ATVA/ball'")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample index (default: 0)")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for ALL datasets")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT,
                        help=f"Output root directory (default: {OUTPUT_ROOT})")
    args = parser.parse_args()

    if args.all:
        datasets = discover_datasets()
        print(f"Found {len(datasets)} datasets, generating sample {args.sample} for each...")
        for ds in datasets:
            print(f"\n[{ds}]")
            analyze_dataset(ds, args.sample, args.output)
    else:
        print(f"[{args.dataset}] sample {args.sample}")
        analyze_dataset(args.dataset, args.sample, args.output)


if __name__ == "__main__":
    main()
