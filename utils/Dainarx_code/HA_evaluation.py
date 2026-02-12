"""
Hybrid Automaton Evaluation Module

This module provides classes and functions for evaluating hybrid automata against ground truth data.
It supports simulation, metric computation, and visualization with multiple plotting modes.
"""

import numpy as np
import os
import sys
from math import *
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
import json
import base64
import io


# Add Dainarx_code directory to sys.path so we can import from src
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Now import from src directly
from src.HybridAutomata import HybridAutomata
from src.Evaluation import Evaluation, max_min_abs_diff


# ============================================================================
# Constants
# ============================================================================
DEFAULT_TIME_STEP = 0.001
DEFAULT_FIGURE_WIDTH = 12
DEFAULT_FIGURE_HEIGHT = 5
DEFAULT_STACKED_HEIGHT = 10
DEFAULT_DPI = 150
DEFAULT_BASE64_DPI = 100
DEFAULT_LINEWIDTH = 2.0
DEFAULT_LINEWIDTH_THICK = 3.5
DEFAULT_LINEWIDTH_THIN = 2.0
DEFAULT_MARKER_SIZE = 6
DEFAULT_MARKER_SIZE_SMALL = 5
DEFAULT_MARKER_INTERVAL = 40  # Show ~40 markers per plot
DEFAULT_FONT_SIZE = 14
DEFAULT_LEGEND_FONT_SIZE = 16
DEFAULT_TITLE_FONT_SIZE = 16


# ============================================================================
# TrajectoryPlotter Class
# ============================================================================
class TrajectoryPlotter:
    """
    Handles plotting of hybrid automaton trajectories with multiple visualization modes.

    Supports three plotting modes:
    - 'single': Plot simulated trajectory only
    - 'overlay': Overlay original and simulated trajectories on the same axes
    - 'stacked': Vertically stack original and simulated trajectories in separate subplots
    """

    def __init__(self,
                 state_data: np.ndarray,
                 input_data: np.ndarray,
                 dt: float = DEFAULT_TIME_STEP,
                 original_state_data: Optional[np.ndarray] = None,
                 original_input_data: Optional[np.ndarray] = None,
                 input_plot: bool = False,
                 title: Optional[str] = None):
        """
        Initialize the trajectory plotter.

        Args:
            state_data: Simulated state data, shape (num_states, num_steps)
            input_data: Simulated input data (empty, 1D, or 2D array)
            dt: Time step size
            original_state_data: Optional ground truth state data
            original_input_data: Optional ground truth input data
            input_plot: Whether to include input data in plots
            title: Optional custom title for single plot mode

        Raises:
            ValueError: If data dimensions are invalid
        """
        # Validate and store state data
        if state_data.ndim != 2:
            raise ValueError("state_data must be a 2D array with shape (num_states, num_steps)")

        self.state_data = state_data
        self.num_states, self.num_steps = state_data.shape

        if self.num_steps == 0:
            raise ValueError("state_data must contain at least one time step")

        # Process and validate input data
        self.input_data = self._process_input_data(input_data, self.num_steps)
        self.num_inputs = self.input_data.shape[0]

        # Store configuration
        self.dt = dt
        self.original_state_data = original_state_data
        self.original_input_data = self._process_input_data(original_input_data, self.num_steps) if original_input_data is not None else None
        self.input_plot = input_plot
        self.title = title

        # Create time array and colormap
        self.time = np.arange(self.num_steps) * dt
        self.cmap = self._get_colormap()

        # Initialize figure
        self.fig = None
        self.axes = None

    @staticmethod
    def _process_input_data(input_data: np.ndarray, expected_steps: int) -> np.ndarray:
        """
        Process input data into consistent 2D format.

        Args:
            input_data: Input data (empty, 1D, or 2D array)
            expected_steps: Expected number of time steps

        Returns:
            Processed input data with shape (num_inputs, num_steps)

        Raises:
            ValueError: If input data has invalid dimensions
        """
        if input_data is None or input_data.size == 0:
            return np.empty((0, expected_steps))
        elif input_data.ndim == 1:
            return input_data.reshape(1, -1)
        elif input_data.ndim == 2:
            return input_data
        else:
            raise ValueError("input_data must be empty, 1D, or 2D array")

    def _get_colormap(self):
        """
        Get colormap for plotting multiple series.

        Returns:
            Colormap object
        """
        total_series = self.num_states + self.num_inputs
        try:
            return plt.cm.get_cmap('tab20', max(total_series, 1))
        except (AttributeError, TypeError):
            # For matplotlib >= 3.7
            import matplotlib as mpl
            return mpl.colormaps['tab20']

    def _plot_trajectory_on_axis(self,
                                  ax,
                                  state_data: np.ndarray,
                                  input_data: np.ndarray,
                                  title: str):
        """
        Plot a single trajectory on the given axis.

        Args:
            ax: Matplotlib axis object
            state_data: State data to plot, shape (num_states, num_steps)
            input_data: Input data to plot (preprocessed)
            title: Title for the subplot
        """
        num_states = state_data.shape[0]
        num_inputs = input_data.shape[0]

        # Plot state variables
        for idx in range(num_states):
            ax.plot(self.time, state_data[idx],
                   label=f"x{idx + 1}",
                   color=self.cmap(idx),
                   linewidth=DEFAULT_LINEWIDTH)

        # Plot input variables if requested
        if self.input_plot and num_inputs > 0:
            for idx in range(num_inputs):
                series_index = num_states + idx
                ax.plot(self.time, input_data[idx],
                       label=f"u{idx + 1}",
                       linestyle='--',
                       color=self.cmap(series_index),
                       linewidth=DEFAULT_LINEWIDTH)

        # Configure axis
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Values')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=DEFAULT_LEGEND_FONT_SIZE)

    def plot_single(self) -> plt.Figure:
        """
        Plot simulated trajectory only.

        Returns:
            Matplotlib figure object
        """
        self.fig, ax = plt.subplots(1, 1,
                                     figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT),
                                     constrained_layout=True)
        plot_title = self.title if self.title is not None else "Simulated Trajectory"
        self._plot_trajectory_on_axis(ax, self.state_data, self.input_data, plot_title)
        self.axes = ax
        return self.fig

    def plot_overlay(self) -> plt.Figure:
        """
        Plot original and simulated trajectories overlaid on the same axes.

        Ground truth data is shown with thick solid lines and circle markers.
        Simulated data is shown with thin dash-dot lines and triangle markers.

        Returns:
            Matplotlib figure object

        Raises:
            ValueError: If original_state_data is not provided
        """
        if self.original_state_data is None:
            raise ValueError("original_state_data must be provided for overlay mode")

        self.fig, ax = plt.subplots(1, 1,
                                     figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT),
                                     constrained_layout=True)

        num_orig_states = self.original_state_data.shape[0]
        marker_interval = max(1, self.num_steps // DEFAULT_MARKER_INTERVAL)

        # Plot original state data (thick solid lines with circle markers)
        for idx in range(num_orig_states):
            ax.plot(self.time, self.original_state_data[idx],
                   label=f"x{idx + 1} (Ground truth)",
                   color=self.cmap(idx),
                   linewidth=DEFAULT_LINEWIDTH_THICK,
                   linestyle='-',
                   marker='o',
                   markersize=DEFAULT_MARKER_SIZE,
                   markevery=marker_interval,
                   alpha=0.6,
                   zorder=1)  # Draw behind simulated data

        # Plot simulated state data (thin dash-dot lines with triangle markers)
        for idx in range(self.num_states):
            ax.plot(self.time, self.state_data[idx],
                   label=f"x{idx + 1} (Simulated)",
                   color=self.cmap(idx),
                   linewidth=DEFAULT_LINEWIDTH_THIN,
                   linestyle='-.',
                   marker='^',
                   markersize=DEFAULT_MARKER_SIZE_SMALL,
                   markevery=marker_interval,
                   alpha=1.0,
                   zorder=2)  # Draw on top

        # Plot input data if requested
        if self.input_plot:
            self._plot_inputs_comparison(ax, marker_interval)

        # Configure axis
        ax.set_xlabel('Time (s)', fontsize=DEFAULT_FONT_SIZE)
        ax.set_ylabel('Values', fontsize=DEFAULT_FONT_SIZE)
        ax.set_title('Comparison: Ground truth vs Simulated Trajectories',
                    fontsize=DEFAULT_TITLE_FONT_SIZE, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=12, ncol=2)

        self.axes = ax
        return self.fig

    def _plot_inputs_comparison(self, ax, marker_interval: int):
        """
        Helper method to plot input comparisons on overlay plot.

        Args:
            ax: Matplotlib axis object
            marker_interval: Interval for showing markers
        """
        # Plot original input data
        if self.original_input_data is not None and self.original_input_data.shape[0] > 0:
            for idx in range(self.original_input_data.shape[0]):
                series_index = self.num_states + idx
                ax.plot(self.time, self.original_input_data[idx],
                       label=f"u{idx + 1} (Ground truth)",
                       linestyle='-',
                       color=self.cmap(series_index),
                       linewidth=DEFAULT_LINEWIDTH_THICK,
                       marker='o',
                       markersize=DEFAULT_MARKER_SIZE,
                       markevery=marker_interval,
                       alpha=0.6,
                       zorder=1)

        # Plot simulated input data
        if self.num_inputs > 0:
            for idx in range(self.num_inputs):
                series_index = self.num_states + idx
                ax.plot(self.time, self.input_data[idx],
                       label=f"u{idx + 1} (Simulated)",
                       linestyle='-.',
                       color=self.cmap(series_index),
                       linewidth=DEFAULT_LINEWIDTH_THIN,
                       marker='^',
                       markersize=DEFAULT_MARKER_SIZE_SMALL,
                       markevery=marker_interval,
                       alpha=1.0,
                       zorder=2)

    def plot_stacked(self) -> plt.Figure:
        """
        Plot original and simulated trajectories in vertically stacked subplots.

        Top subplot: Simulated trajectory
        Bottom subplot: Ground truth trajectory

        Returns:
            Matplotlib figure object

        Raises:
            ValueError: If original_state_data is not provided
        """
        if self.original_state_data is None:
            raise ValueError("original_state_data must be provided for stacked mode")

        self.fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1,
            figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_STACKED_HEIGHT),
            constrained_layout=True,
            sharex=True
        )

        # Top: Simulated trajectory
        self._plot_trajectory_on_axis(ax_top, self.state_data, self.input_data,
                                      "Simulated Trajectory")

        # Bottom: Ground truth trajectory
        original_input = self.original_input_data if self.original_input_data is not None else np.array([])
        self._plot_trajectory_on_axis(ax_bottom, self.original_state_data, original_input,
                                      "Ground truth Trajectory")

        self.axes = (ax_top, ax_bottom)
        return self.fig

    def plot(self, mode: str = "single") -> plt.Figure:
        """
        Generate plot based on specified mode.

        Args:
            mode: Plotting mode - "single", "overlay", or "stacked"

        Returns:
            Matplotlib figure object

        Raises:
            ValueError: If mode is invalid
        """
        if mode == "single":
            # single: Plot simulated trajectory only.
            return self.plot_single()
        elif mode == "overlay":
            # overlay: Plot simulated and ground truth on the same axis.
            return self.plot_overlay()
        elif mode == "stacked":
            # stacked: Plot simulated (top) and ground truth (bottom) in separate subplots.
            return self.plot_stacked()
        else:
            raise ValueError(f"Invalid plot mode: {mode}. Must be 'single', 'overlay', or 'stacked'")

    def save(self, save_path: str, dpi: int = DEFAULT_DPI):
        """
        Save the current figure to file.

        Args:
            save_path: Path to save the figure
            dpi: Resolution in dots per inch
        """
        if self.fig is None:
            raise RuntimeError("No figure to save. Call plot() first.")

        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.fig.savefig(save_path, dpi=dpi)

    def to_base64(self, dpi: int = DEFAULT_BASE64_DPI) -> str:
        """
        Convert the current figure to base64-encoded PNG string.

        Args:
            dpi: Resolution in dots per inch

        Returns:
            Base64-encoded PNG image string
        """
        if self.fig is None:
            raise RuntimeError("No figure to convert. Call plot() first.")

        buffer = io.BytesIO()
        self.fig.savefig(buffer, format='png', dpi=dpi)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()

        return base64_image

    def close(self):
        """Close the figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


# ============================================================================
# HAEvaluator Class
# ============================================================================
class HAEvaluator:
    """
    Evaluates a hybrid automaton against ground truth data.

    This class handles:
    - Loading ground truth data from NPZ files
    - Simulating hybrid automaton dynamics
    - Computing evaluation metrics (absolute and normalized errors)
    - Generating comparison visualizations
    """

    def __init__(self,
                 ha_dict: Dict[str, Any],
                 npz_file_path: str,
                 dt: Optional[float] = None,
                 total_time: float = 10.0):
        """
        Initialize the evaluator.

        Args:
            ha_dict: Dictionary containing 'automaton' and 'config' keys
            npz_file_path: Path to NPZ file with ground truth data
            dt: Time step (if None, uses value from ha_dict['config']['dt'])
            total_time: Total simulation time in seconds
        """
        self.ha_dict = ha_dict
        self.npz_file_path = npz_file_path
        self.total_time = total_time

        # Extract time step from config or use provided value
        self.dt = dt if dt is not None else ha_dict['config']['dt']

        # Initialize storage for results
        self.ground_truth = None
        self.simulation_results = None
        self.metrics = None

        # Create hybrid automaton from JSON specification
        self.ha_system = HybridAutomata.from_json(ha_dict["automaton"])

    def load_ground_truth(self) -> Dict[str, np.ndarray]:
        """
        Load ground truth data from NPZ file.

        Returns:
            Dictionary containing 'state', 'input', 'mode', 'change_points'
        """
        npz_file = np.load(self.npz_file_path)

        self.ground_truth = {
            'state': npz_file['state'],
            'input': npz_file['input'],
            'mode': npz_file['mode'] if 'mode' in npz_file else None,
            'change_points': npz_file['change_points'].tolist() if 'change_points' in npz_file else None
        }

        return self.ground_truth

    def _prepare_initial_state(self) -> Dict[str, Any]:
        """
        Prepare initial state dictionary for simulation.

        Returns:
            Initial state dictionary with mode, state variables, and input
        """
        if self.ground_truth is None:
            self.load_ground_truth()

        state_data_npz = self.ground_truth['state']
        input_data_npz = self.ground_truth['input']

        # Extract initial state from first column
        init_state = state_data_npz[:, 0].tolist()  # Convert to list to handle both single and multiple variables
        if not isinstance(init_state, list):
            init_state = [init_state]

        # Parse variable names from automaton specification
        var_names = [v.strip() for v in self.ha_dict['automaton']['var'].split(',')]

        # Build initial state dictionary
        init_state_dict = {'mode': 1}

        if len(var_names) == 1:
            # Single variable case (e.g., "x")
            init_state_dict[var_names[0]] = init_state
        else:
            # Multiple variables case (e.g., "x1, x2, x3, x4")
            for i, var_name in enumerate(var_names):
                init_state_dict[var_name] = [init_state[i]] if i < len(init_state) else [0.0]

        # Add input data (use full time series, not function expression)
        # Only add if 'input' key exists and is not empty
        input_var_name = self.ha_dict['automaton'].get('input', '')
        if input_var_name:  # Only add if input variable name is non-empty
            init_state_dict[input_var_name] = input_data_npz

        return init_state_dict

    def simulate(self) -> Dict[str, np.ndarray]:
        """
        Run hybrid automaton simulation.

        Returns:
            Dictionary containing 'state', 'input', 'mode', 'change_points'
        """
        # Prepare initial state
        init_state_dict = self._prepare_initial_state()

        # Initialize simulation
        state_data = []
        mode_data = []
        input_data = []
        change_points = [0]

        self.ha_system.reset(init_state_dict, dt=self.dt)

        # Run simulation
        current_time = 0.0
        time_step_idx = 0

        while current_time < self.total_time:
            current_time += self.dt
            time_step_idx += 1

            state, mode, switched = self.ha_system.next(self.dt)

            state_data.append(state)
            mode_data.append(mode)
            input_data.append(self.ha_system.getInput())

            if switched:
                change_points.append(time_step_idx)

        # Mark end of simulation as final change point
        change_points.append(time_step_idx)

        # Convert to numpy arrays and transpose to (num_variables, num_steps) format
        self.simulation_results = {
            'state': np.transpose(np.array(state_data)),
            'input': np.transpose(np.array(input_data)),
            'mode': np.array(mode_data),
            'change_points': change_points
        }

        return self.simulation_results

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics using the Evaluation class.

        Returns:
            Dictionary containing computed metrics from Evaluation class
        """
        if self.ground_truth is None:
            self.load_ground_truth()

        if self.simulation_results is None:
            self.simulate()

        # Extract data
        sim_state = self.simulation_results['state']
        sim_mode = self.simulation_results['mode']
        sim_change_points = self.simulation_results['change_points']

        gt_state = self.ground_truth['state']
        gt_mode = self.ground_truth['mode']
        gt_change_points = self.ground_truth['change_points']

        # Truncate to shorter length for fair comparison
        # Note: sim[i] corresponds to GT[i+1] due to timing offset
        # (sim uses GT[0] as initial state, so sim[0] is at t=2*dt while GT[0] is at t=dt)
        min_len = min(sim_state.shape[1], gt_state.shape[1] - 1)
        sim_state_truncated = sim_state[:, :min_len]
        gt_state_truncated = gt_state[:, 1:min_len+1]  # Offset by 1

        # Use Evaluation class to compute metrics
        evaluator = Evaluation(name="HA_Single_Trajectory_Evaluation")

        evaluator.submit(
            fit_mode=[sim_mode[:min_len]],
            fit_data=[sim_state_truncated],
            gt_mode=[gt_mode[1:min_len+1]] if gt_mode is not None else [[1] * min_len],  # Offset by 1
            gt_data=[gt_state_truncated],
            chp=[sim_change_points],
            gt_chp=[gt_change_points] if gt_change_points is not None else [[0, min_len]],
            mode_num=len(set(sim_mode)),
            gt_mode_num=len(set(gt_mode)) if gt_mode is not None else 1,
            dt=self.dt
        )

        eval_results = evaluator.calc()

        # Assemble results
        self.metrics = {
            'tc': float(eval_results['tc']) if eval_results['tc'] is not None else None,
            'train_tc': float(eval_results['train_tc']),
            'max_diff': float(eval_results['max_diff']) if eval_results['max_diff'] is not None else None,
            'mean_diff': float(eval_results['mean_diff']) if eval_results['mean_diff'] is not None else None,
            'clustering_error': int(eval_results['clustering_error']),
            'simulated_data': self.simulation_results,
            'ground_truth_data': self.ground_truth
        }

        return self.metrics

    def plot(self,
             plot_mode: str = "single",
             save_path: Optional[str] = None,
             input_plot: bool = False,
             return_base64: bool = False) -> Optional[str]:
        """
        Generate visualization of trajectories.

        Args:
            plot_mode: Plotting mode - "single", "overlay", or "stacked"
            save_path: Path to save the figure (optional)
            input_plot: Whether to include input plots
            return_base64: If True, return base64-encoded PNG

        Returns:
            Base64-encoded PNG string if return_base64=True, otherwise None
        """
        if self.ground_truth is None:
            self.load_ground_truth()

        if self.simulation_results is None:
            self.simulate()

        # Create plotter
        plotter = TrajectoryPlotter(
            state_data=self.simulation_results['state'],
            input_data=self.simulation_results['input'],
            dt=self.dt,
            original_state_data=self.ground_truth['state'],
            original_input_data=self.ground_truth['input'],
            input_plot=input_plot
        )

        # Generate plot
        plotter.plot(mode=plot_mode)

        # Save if requested
        if save_path is not None:
            plotter.save(save_path)

        # Convert to base64 if requested
        result = None
        if return_base64:
            result = plotter.to_base64()

        # Clean up
        plotter.close()

        return result

    def __call__(self,
                 plot_mode: str = "overlay",
                 save_path: Optional[str] = None,
                 print_metrics: bool = True) -> Dict[str, Any]:
        """
        Convenience method to directly call evaluator and display results.

        This method makes HAEvaluator callable, providing an easy interface for
        evaluation with automatic plot generation and metric display.

        Args:
            plot_mode: Plotting mode - "single", "overlay", or "stacked"
            save_path: Optional path to save the plot
            print_metrics: Whether to print formatted metrics to console

        Returns:
            Dictionary containing all evaluation metrics and results

        Example:
            >>> evaluator = HAEvaluator(ha_dict, 'data/test.npz')
            >>> results = evaluator()  # Shows plot and prints metrics
            >>> results = evaluator(plot_mode="stacked", save_path="output.png")
        """
        # Load ground truth
        self.load_ground_truth()

        # Run simulation
        self.simulate()

        # Compute metrics
        self.compute_metrics()

        # Generate plot
        if self.ground_truth is None:
            self.load_ground_truth()

        if self.simulation_results is None:
            self.simulate()

        # Create plotter
        plotter = TrajectoryPlotter(
            state_data=self.simulation_results['state'],
            input_data=self.simulation_results['input'],
            dt=self.dt,
            original_state_data=self.ground_truth['state'],
            original_input_data=self.ground_truth['input'],
            input_plot=False
        )

        # Generate plot
        plotter.plot(mode=plot_mode)
        plot_base64 = plotter.to_base64()
        # Save if requested
        if save_path is not None:
            # Convert relative path to absolute path based on this file's location
            # if not os.path.isabs(save_path):
            #     save_path = os.path.join(_dainarx_code_dir, save_path)
            plotter.save(save_path)
            if print_metrics:
                print(f"Plot saved to: {save_path}")

        # Clean up
        plotter.close()

        # Print metrics if requested
        metrics_text = self._print_formatted_metrics(print_to_console=print_metrics)

        return metrics_text['text'], plot_base64

    def _print_formatted_metrics(self, print_to_console: bool = True) -> dict:
        """
        Format evaluation metrics and optionally print to console.

        Args:
            print_to_console: If True, print formatted metrics to console.

        Returns:
            dict: Formatted metrics with the following keys:
                - 'formatted': Dictionary with formatted string values
                - 'text': Complete formatted text output
        """
        # Build formatted output lines
        lines = []

        # Build formatted values dictionary
        formatted = {}
        '''
        4. **Metrics Interpretation**:
        - The evaluation metrics show how well the HA matches ground truth data
        - `tc` (change-point error): How accurately mode switches are detected (lower is better)
        - `max_diff` / `mean_diff`: Maximum and mean state trajectory differences (lower is better)
        - `clustering_error`: How well modes are classified (lower is better)
        
        '''

        if self.metrics['tc'] is not None:
            formatted['tc'] = f"{self.metrics['tc']:.6f} seconds"
            lines.append(f"  TC (Change-Point Error):{formatted['tc']}")
        else:
            formatted['tc'] = "N/A"
            lines.append(f"  TC (Change-Point Error):N/A")

        if self.metrics['max_diff'] is not None:
            formatted['max_diff'] = f"{self.metrics['max_diff']:.6f}"
            formatted['mean_diff'] = f"{self.metrics['mean_diff']:.6f}"
            lines.append(f"  Max Difference:{formatted['max_diff']}")
            lines.append(f"  Mean Difference:{formatted['mean_diff']}")
        else:
            formatted['max_diff'] = "N/A"
            formatted['mean_diff'] = "N/A"
            lines.append(f"  Max Difference:N/A")
            lines.append(f"  Mean Difference:N/A")


        # Join all lines into formatted text
        formatted_text = "\n".join(lines)

        # Print to console if requested
        if print_to_console:
            print(formatted_text)

        # Return structured result
        return {
            'formatted': formatted,
            'text': formatted_text
        }



# ============================================================================
# Main Execution Block (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test data: Single mode Duffing oscillator
    data1 = {
        "automaton": {
            "var": "x1",
            "input": "u1",
            "mode": [
                {
                    "id": 1,
                    "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] - 0.5 * x1[0]**3 + u1"
                }
            ],
            "edge": []
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "dim": 2,
            "other_items": ""
        }
    }
    data2 = {
        "automaton": {
            "var": "x1, x2",
            "input": "",
            "mode": [
            {
                "id": 1,
                "eq": "x1[1] = x2[0], x2[1] = -9.8"
            }
            ],
            "edge": [
            {
                "direction": "1 -> 1",
                "condition": "x1 <= 0 and x2 < 0",
                "reset": {
                "x1": [
                    "0"
                ],
                "x2": [
                    "-0.9 * x2[0]"
                ]
                }
            }
            ]
        },
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "order": 1,
            "self_loop": True
        }
        }
    data=json.load(open('utils/Dainarx_code/automata/ATVA/ball.json', 'r'))
    # print(data)
    


    # Create evaluator using the new HAEvaluator class
    print("Testing HAEvaluator with __call__() method...")
    print("=" * 80)

    evaluator = HAEvaluator(
        ha_dict=data,
        npz_file_path='data_all/ATVA/ball_g/ground_truth_0.npz',
        dt=0.001,
        total_time=10.0
    )
    ## Because the ball system only has one mode, so the TC is always 0

    # Mode 1: Single plot (simulated only) with metrics
    print("\n1. Generating single plot (simulated only)...")
    metrics_text_str, plot_base64 = evaluator(
        plot_mode="single",
        save_path='data_duffing_evaluation/output_single.png',
        print_metrics=True
    )
    print("Metrics text:", metrics_text_str)
    print("Metrics:", evaluator.metrics)

    # Mode 2: Overlay comparison
    print("\n2. Generating overlay comparison plot...")
    metrics_text, plot_base64 = evaluator(
        plot_mode="overlay",
        save_path='data_duffing_evaluation/output_overlay.png',
        print_metrics=False
    )

    # Mode 3: Stacked comparison
    print("\n3. Generating stacked comparison plot...")
    metrics_text, plot_base64 = evaluator(
        plot_mode="stacked",
        save_path='data_duffing_evaluation/output_stacked.png',
        print_metrics=False
    )
