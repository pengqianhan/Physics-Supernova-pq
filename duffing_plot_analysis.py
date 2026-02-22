"""
Duffing oscillator ground truth analysis:
- Trajectory colored by mode
- Position, velocity, acceleration with mode transition markers
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("data_all/non_linear/duffing_g/ground_truth_0.npz")
state = data["state"]  # (1, 10001)
mode = data["mode"]    # (10001,)
inp = data["input"]    # (1, 10001)
change_points = data["change_points"]

x1 = state[0]          # position
u1 = inp[0]            # input

dt = 0.001
total_time = (len(x1) - 1) * dt
t = np.linspace(0, total_time, len(x1))

# Compute derivatives via finite differences
x1_dot = np.gradient(x1, dt)       # velocity
x1_ddot = np.gradient(x1_dot, dt)  # acceleration

# Mode transition indices (interior change points, excluding 0 and last)
transitions = change_points[(change_points > 0) & (change_points < len(x1))]

# Color map for modes
mode_colors = {1: "tab:blue", 2: "tab:red"}
unique_modes = np.unique(mode)

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# --- Helper to plot signal colored by mode with transition lines ---
def plot_colored_by_mode(ax, t, y, ylabel, label, transitions, mode, mode_colors):
    # Plot segments by mode
    segments = np.split(np.arange(len(t)), np.where(np.diff(mode) != 0)[0] + 1)
    for seg in segments:
        if len(seg) == 0:
            continue
        m = mode[seg[0]]
        color = mode_colors.get(m, "gray")
        ax.plot(t[seg], y[seg], color=color, linewidth=0.8)
    # Transition lines
    for cp in transitions:
        if cp < len(t):
            ax.axvline(t[cp], color="black", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=12)
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=mode_colors[m], lw=2, label=f"Mode {m}") for m in sorted(mode_colors)]
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1, alpha=0.5, label="Mode transition"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

# 1) Position
plot_colored_by_mode(axes[0], t, x1, "Position", "x1", transitions, mode, mode_colors)
axes[0].set_title("Duffing Oscillator — Ground Truth (sample 0)", fontsize=14, fontweight="bold")

# 2) Velocity
plot_colored_by_mode(axes[1], t, x1_dot, "Velocity (dx1/dt)", "x1_dot", transitions, mode, mode_colors)

# 3) Acceleration
plot_colored_by_mode(axes[2], t, x1_ddot, "Acceleration (d²x1/dt²)", "x1_ddot", transitions, mode, mode_colors)

# 4) Input
plot_colored_by_mode(axes[3], t, u1, "Input (u1)", "u1", transitions, mode, mode_colors)

axes[3].set_xlabel("Time (s)", fontsize=12)

plt.tight_layout()
plt.savefig("duffing_ground_truth_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: duffing_ground_truth_analysis.png")
