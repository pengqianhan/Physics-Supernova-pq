"""
Ground truth derivative analysis for hybrid automaton datasets.
Plots each state variable's position, velocity, acceleration colored by mode,
plus input signals, to visualize whether mode transitions correspond to
derivative discontinuities.

Usage:
    python data_plot_analysis_groundtruth.py                           # default: duffing, sample 0
    python data_plot_analysis_groundtruth.py --dataset non_linear/duffing --sample 0
    python data_plot_analysis_groundtruth.py --dataset ATVA/ball --sample 3
    python data_plot_analysis_groundtruth.py --all                     # generate for ALL datasets, sample 0
    python data_plot_analysis_groundtruth.py --all --sample 5
"""

import argparse
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_ROOT = "data_all"
AUTOMATA_ROOT = "utils/Dainarx_code/automata"
OUTPUT_ROOT = "analysis_plots"

# Distinct colors for up to 10 modes
MODE_PALETTE = [
    "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


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


def build_mode_colors(unique_modes):
    return {m: MODE_PALETTE[i % len(MODE_PALETTE)] for i, m in enumerate(sorted(unique_modes))}


def plot_colored_by_mode(ax, t, y, ylabel, transitions, mode, mode_colors):
    """Plot a signal colored by mode with transition markers."""
    segments = np.split(np.arange(len(t)), np.where(np.diff(mode) != 0)[0] + 1)
    for seg in segments:
        if len(seg) == 0:
            continue
        m = mode[seg[0]]
        ax.plot(t[seg], y[seg], color=mode_colors.get(m, "gray"), linewidth=0.8)
    for cp in transitions:
        if cp < len(t):
            ax.axvline(t[cp], color="black", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=11)
    handles = [Line2D([0], [0], color=c, lw=2, label=f"Mode {m}")
               for m, c in sorted(mode_colors.items())]
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1, alpha=0.5, label="Transition"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def analyze_dataset(dataset, sample_idx, output_root):
    """Generate the derivative analysis plot for one dataset + sample."""
    gt_dir = os.path.join(DATA_ROOT, f"{dataset}_g")
    npz_path = os.path.join(gt_dir, f"ground_truth_{sample_idx}.npz")
    if not os.path.exists(npz_path):
        print(f"  [SKIP] {npz_path} not found")
        return

    data = np.load(npz_path)
    state = data["state"]       # (num_vars, num_steps)
    mode = data["mode"]         # (num_steps,)
    inp = data["input"]         # (num_inputs, num_steps) or (num_steps,)
    change_points = data["change_points"]

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

    # Pad var_names if fewer than num_vars
    while len(var_names) < num_vars:
        var_names.append(f"x{len(var_names)+1}")

    t = np.arange(num_steps) * dt
    transitions = change_points[(change_points > 0) & (change_points < num_steps)]
    unique_modes = np.unique(mode)
    mode_colors = build_mode_colors(unique_modes)

    # Layout: for each state var -> 3 rows (pos, vel, acc), plus input rows
    input_rows = num_inputs if has_input else 0
    total_rows = num_vars * 3 + input_rows
    fig_height = max(8, total_rows * 2.8)
    fig, axes = plt.subplots(total_rows, 1, figsize=(14, fig_height), sharex=True)
    if total_rows == 1:
        axes = [axes]

    system_name = dataset.split("/")[-1]
    fig.suptitle(f"{system_name} — Ground Truth (sample {sample_idx})",
                 fontsize=14, fontweight="bold", y=0.998)

    row = 0
    for vi in range(num_vars):
        x = state[vi]
        x_dot = np.gradient(x, dt)
        x_ddot = np.gradient(x_dot, dt)
        vname = var_names[vi]

        plot_colored_by_mode(axes[row], t, x,
                             f"{vname}\n(position)", transitions, mode, mode_colors)
        row += 1
        plot_colored_by_mode(axes[row], t, x_dot,
                             f"d{vname}/dt\n(velocity)", transitions, mode, mode_colors)
        row += 1
        plot_colored_by_mode(axes[row], t, x_ddot,
                             f"d\u00b2{vname}/dt\u00b2\n(accel.)", transitions, mode, mode_colors)
        row += 1

    for ui in range(input_rows):
        u = inp[ui]
        plot_colored_by_mode(axes[row], t, u,
                             f"u{ui+1}\n(input)", transitions, mode, mode_colors)
        row += 1

    axes[-1].set_xlabel("Time (s)", fontsize=12)

    # Save with matching directory structure
    out_dir = os.path.join(output_root, dataset + "_g")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ground_truth_{sample_idx}_analysis.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def discover_datasets():
    """Find all datasets that have a _g ground truth directory."""
    datasets = []
    for gt_dir in sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*_g"), recursive=True)):
        rel = os.path.relpath(gt_dir, DATA_ROOT)  # e.g. "non_linear/duffing_g"
        dataset = rel.rsplit("_g", 1)[0]           # e.g. "non_linear/duffing"
        datasets.append(dataset)
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Ground truth derivative analysis")
    parser.add_argument("--dataset", type=str, default="non_linear/duffing",
                        help="Dataset path relative to data_all, e.g. 'non_linear/duffing', 'ATVA/ball'")
    parser.add_argument("--sample", type=int, default=0,
                        help="Ground truth sample index (default: 0)")
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
