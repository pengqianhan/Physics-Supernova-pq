"""
Direct Python Class-Based Hybrid Automaton Simulator

This module provides direct simulation and evaluation of Python class-based HA specifications
without requiring JSON conversion. Inspired by SR-Scientist's direct code execution approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
import warnings


class HybridAutomatonSimulator:
    """
    Direct simulator for Python class-based Hybrid Automata.

    Executes the HybridAutomaton class methods directly for simulation,
    avoiding JSON conversion overhead.
    """

    def __init__(self, ha_instance, dt: float = 0.001, total_time: float = 10.0):
        """
        Initialize simulator with HA instance.

        Args:
            ha_instance: Instance of HybridAutomaton class
            dt: Integration time step
            total_time: Total simulation duration
        """
        self.ha = ha_instance
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)

    def simulate(self, initial_state: np.ndarray, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the hybrid automaton.

        Args:
            initial_state: Initial state values [shape: (num_vars,)]
            input_data: Input time series [shape: (num_inputs, num_steps)] or (num_steps,)

        Returns:
            state_trajectory: State evolution [shape: (num_vars, num_steps)]
            mode_sequence: Active mode at each step [shape: (num_steps,)]
        """
        # Handle input data shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        num_vars = len(initial_state)
        num_inputs = input_data.shape[0]

        # Initialize storage
        state_trajectory = np.zeros((num_vars, self.num_steps))
        mode_sequence = np.zeros(self.num_steps, dtype=int)

        # Parse variable and input names from HA
        var_names = [v.strip() for v in self.ha.var.split(',')]
        input_names = [i.strip() for i in self.ha.input.split(',')] if self.ha.input else []

        # Initialize state dict (for higher-order ODEs, store derivatives)
        # For order=2: x_dict = {'x1': [x1_value, x1_dot]}
        order = getattr(self.ha, 'order', 1)
        x_dict = {}
        for i, var_name in enumerate(var_names):
            x_dict[var_name] = [initial_state[i]] + [0.0] * (order - 1)

        # Get initial mode
        current_mode = 1

        # Simulation loop
        for step in range(self.num_steps):
            # Build input dict
            u_dict = {}
            for i, input_name in enumerate(input_names):
                if i < num_inputs and step < input_data.shape[1]:
                    u_dict[input_name] = input_data[i, step]
                else:
                    u_dict[input_name] = 0.0

            # Check for mode transitions
            new_mode = self._check_transitions(current_mode, x_dict, u_dict)

            # Apply reset map if mode changed
            if new_mode != current_mode:
                self.ha.reset_map(current_mode, new_mode, x_dict, u_dict)
                current_mode = new_mode

            # Store current state and mode
            for i, var_name in enumerate(var_names):
                state_trajectory[i, step] = x_dict[var_name][0]
            mode_sequence[step] = current_mode

            # Integrate dynamics (Euler method for simplicity)
            if step < self.num_steps - 1:
                self._integrate_step(current_mode, x_dict, u_dict)

        return state_trajectory, mode_sequence

    def _check_transitions(self, current_mode: int, x_dict: Dict, u_dict: Dict) -> int:
        """Check all possible transitions from current mode."""
        num_modes = self.ha.num_modes()

        # Check transitions to all other modes
        for target_mode in range(1, num_modes + 1):
            if target_mode != current_mode:
                if self.ha.guard_condition(current_mode, target_mode, x_dict, u_dict):
                    return target_mode

        # Check self-loops
        if self.ha.guard_condition(current_mode, current_mode, x_dict, u_dict):
            return current_mode

        return current_mode

    def _integrate_step(self, mode_id: int, x_dict: Dict, u_dict: Dict):
        """
        Integrate one time step using Euler method.

        Parses the ODE equation string and updates state derivatives.
        """
        # Get ODE equation string from mode dynamics
        eq_str = self.ha.mode_dynamics(mode_id, x_dict, u_dict)

        # Parse equation: "var[k] = expression"
        # Example: "x1[2] = -0.5*x1[1] - 5*x1[0] + u1"
        try:
            lhs, rhs = eq_str.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Extract variable name and derivative order
            var_name = lhs.split('[')[0].strip()
            deriv_order = int(lhs.split('[')[1].split(']')[0])

            # Build evaluation namespace
            eval_namespace = {}

            # Add state variables to namespace
            for var, values in x_dict.items():
                for k, val in enumerate(values):
                    eval_namespace[f'{var}_{k}'] = val
                # Also add direct indexing (for eval)
                eval_namespace[var] = values

            # Add input variables (both as dict and individual values)
            eval_namespace['u'] = u_dict  # For u['u1'] notation
            for inp_name, inp_val in u_dict.items():
                eval_namespace[inp_name] = inp_val  # For direct u1 notation

            # Replace x[k] notation with x_k for evaluation
            eval_rhs = rhs
            for var in x_dict.keys():
                for k in range(len(x_dict[var])):
                    eval_rhs = eval_rhs.replace(f'{var}[{k}]', f'{var}_{k}')

            # Evaluate RHS
            derivative_value = eval(eval_rhs, {"__builtins__": {}}, eval_namespace)

            # Update state using Euler method
            # For order=1: x[0] += dt * x[1]
            # For order=2: x[0] += dt * x[1], x[1] += dt * x[2]
            for k in range(deriv_order - 1):
                x_dict[var_name][k] += self.dt * x_dict[var_name][k + 1]

            # Update highest derivative
            x_dict[var_name][deriv_order - 1] = derivative_value

        except Exception as e:
            warnings.warn(f"Failed to parse/evaluate equation '{eq_str}': {e}")


class PythonClassHAEvaluator:
    """
    Evaluator for Python class-based Hybrid Automata.

    Directly simulates and compares against ground truth without JSON conversion.
    """

    def __init__(self, ha_instance, npz_file_path: str, dt: float = 0.001, total_time: float = 10.0):
        """
        Initialize evaluator.

        Args:
            ha_instance: Instance of HybridAutomaton class
            npz_file_path: Path to ground truth NPZ file
            dt: Integration time step
            total_time: Total simulation duration
        """
        self.ha = ha_instance
        self.npz_file_path = npz_file_path
        self.dt = dt
        self.total_time = total_time

        # Load ground truth
        self.gt_data = np.load(npz_file_path, allow_pickle=True)

        # Create simulator
        self.simulator = HybridAutomatonSimulator(ha_instance, dt, total_time)

        # Storage for results
        self.sim_state = None
        self.sim_mode = None
        self.metrics = {}

    def run_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation using ground truth initial conditions and inputs."""
        # Get initial state
        gt_state = self.gt_data['state']
        initial_state = gt_state[:, 0]

        # Get input data
        if 'input' in self.gt_data:
            input_data = self.gt_data['input']
        else:
            # No input, use zeros
            num_steps = gt_state.shape[1]
            input_data = np.zeros((1, num_steps))

        # Simulate
        self.sim_state, self.sim_mode = self.simulator.simulate(initial_state, input_data)

        return self.sim_state, self.sim_mode

    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics comparing simulation to ground truth."""
        if self.sim_state is None:
            self.run_simulation()

        gt_state = self.gt_data['state']

        # Truncate to minimum length
        min_len = min(self.sim_state.shape[1], gt_state.shape[1])
        sim_truncated = self.sim_state[:, :min_len]
        gt_truncated = gt_state[:, :min_len]

        # Compute state errors
        diff = sim_truncated - gt_truncated
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))

        # Compute mode switch error (TC - change-point error)
        tc = 0.0
        if 'change_points' in self.gt_data and 'mode' in self.gt_data:
            gt_mode = self.gt_data['mode'][:min_len]
            sim_mode_truncated = self.sim_mode[:min_len]

            # Find switch points in both
            gt_switches = np.where(np.diff(gt_mode) != 0)[0]
            sim_switches = np.where(np.diff(sim_mode_truncated) != 0)[0]

            # Compute time error for each GT switch
            if len(gt_switches) > 0 and len(sim_switches) > 0:
                errors = []
                for gt_sw in gt_switches:
                    # Find closest sim switch
                    closest_sim = sim_switches[np.argmin(np.abs(sim_switches - gt_sw))]
                    time_error = abs(closest_sim - gt_sw) * self.dt
                    errors.append(time_error)
                tc = np.mean(errors) if errors else 0.0
            else:
                tc = 0.0 if len(gt_switches) == 0 and len(sim_switches) == 0 else 1.0

        self.metrics = {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rmse': rmse,
            'tc': tc
        }

        return self.metrics

    def plot(self, save_path: Optional[str] = None, show_plot: bool = True) -> str:
        """Generate comparison plot."""
        if self.sim_state is None:
            self.run_simulation()

        gt_state = self.gt_data['state']
        num_vars = gt_state.shape[0]

        fig, axes = plt.subplots(num_vars, 1, figsize=(12, 3 * num_vars))
        if num_vars == 1:
            axes = [axes]

        # Parse variable names
        var_names = [v.strip() for v in self.ha.var.split(',')]

        # Plot each variable
        min_len = min(self.sim_state.shape[1], gt_state.shape[1])
        time = np.arange(min_len) * self.dt

        for i, (ax, var_name) in enumerate(zip(axes, var_names)):
            ax.plot(time, gt_state[i, :min_len], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(time, self.sim_state[i, :min_len], 'r--', label='Simulation', linewidth=2, alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(var_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{var_name} - Max Error: {np.max(np.abs(self.sim_state[i, :min_len] - gt_state[i, :min_len])):.4f}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path or ""

    def __call__(self, plot_mode: str = 'overlay', save_path: Optional[str] = None,
                 print_metrics: bool = True, show_plot: bool = False) -> Tuple[str, Optional[str]]:
        """
        Run full evaluation pipeline.

        Returns:
            metrics_text: Formatted metrics string
            plot_path: Path to saved plot (if save_path provided)
        """
        # Run simulation
        self.run_simulation()

        # Compute metrics
        metrics = self.compute_metrics()

        # Format metrics
        metrics_text = f"""
Evaluation Metrics:
- Max Difference (max_diff): {metrics['max_diff']:.6f}
- Mean Difference (mean_diff): {metrics['mean_diff']:.6f}
- RMSE: {metrics['rmse']:.6f}
- Change-Point Error (TC): {metrics['tc']:.6f}
"""

        if print_metrics:
            print(metrics_text)

        # Generate plot
        plot_path = self.plot(save_path=save_path, show_plot=show_plot)

        return metrics_text, plot_path if save_path else None
