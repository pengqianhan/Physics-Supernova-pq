"""
Parameter optimization for Python class-based Hybrid Automaton specifications.

This module uses gradient-free optimization (Simulated Annealing, Hill Climbing) to tune
the numerical parameters in HA classes to minimize error against ground truth trajectories.
"""

import sys
import os
import numpy as np
from typing import Tuple, Dict, Any, Optional
import traceback
import re

# Add Dainarx code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Dainarx_code'))

from gradient_free_optimizers import SimulatedAnnealingOptimizer, HillClimbingOptimizer
from utils.ha_class_to_json import convert_python_class_to_json
from utils.ha_class_validator import extract_initial_params_from_class
from HA_evaluation import HAEvaluator


def auto_scale_search_space(initial_params: list, scale_factor: float = 10.0) -> dict:
    """
    Automatically create search space for optimization based on initial parameter values.

    Args:
        initial_params: Initial parameter guesses
        scale_factor: Multiplier for search range (default: ±10x)

    Returns:
        Dictionary mapping parameter names to numpy linspace arrays
        Format: {'p0': np.linspace(lower, upper, 1000), 'p1': ...}

    Examples:
        >>> auto_scale_search_space([0.5, -5.0, 1.0])
        {'p0': array([...]), 'p1': array([...]), 'p2': array([...])}
    """
    search_space = {}

    for i, p_init in enumerate(initial_params):
        param_name = f'p{i}'

        if abs(p_init) < 1e-6:
            # Near-zero parameter: use symmetric range around zero
            search_space[param_name] = np.linspace(-1.0, 1.0, 1000)
        else:
            # Non-zero parameter: use ±scale_factor * initial value
            lower = min(p_init * scale_factor, p_init / scale_factor)
            upper = max(p_init * scale_factor, p_init / scale_factor)
            search_space[param_name] = np.linspace(lower, upper, 1000)

    return search_space


def inject_params_into_class(class_code: str, optimized_params: np.ndarray) -> str:
    """
    Inject optimized parameter values into the class code.

    Replaces the self.params = [...] assignment in __init__ with optimized values.

    Args:
        class_code: Original Python class code
        optimized_params: Optimized parameter array

    Returns:
        Updated class code with optimized params

    Examples:
        >>> code = "self.params = [1.0, 2.0, 3.0]"
        >>> inject_params_into_class(code, np.array([1.5, 2.5, 3.5]))
        "self.params = [1.5, 2.5, 3.5]"
    """
    # Convert params to list format
    params_str = str(optimized_params.tolist())

    # Pattern to match self.params = [...] assignment
    pattern = r'(self\.params\s*=\s*)\[[^\]]*\]'

    # Replace with optimized params
    updated_code = re.sub(pattern, rf'\1{params_str}', class_code, count=1)

    return updated_code


def optimize_ha_params(
    class_code: str,
    npz_file_path: str,
    n_iter: int = 100,
    optimizer_type: str = 'simulated_annealing',
    dt: float = 0.001,
    total_time: float = 10.0,
    verbose: bool = False
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Optimize parameters in HA Python class using gradient-free optimization.

    This function:
    1. Extracts initial parameters from the class
    2. Defines objective function (converts class → JSON → simulate → compute error)
    3. Runs gradient-free optimization (Simulated Annealing or Hill Climbing)
    4. Injects optimized parameters back into class code

    Args:
        class_code: Python class code string
        npz_file_path: Path to ground truth NPZ data
        n_iter: Number of optimization iterations (default: 100)
        optimizer_type: 'simulated_annealing' or 'hill_climbing' (default: 'simulated_annealing')
        dt: Time step for simulation (default: 0.001)
        total_time: Total simulation time (default: 10.0)
        verbose: Print optimization progress (default: False)

    Returns:
        Tuple of (optimized_class_code, optimized_params, metrics):
        - optimized_class_code: Updated class with optimized params
        - optimized_params: Numpy array of optimized parameter values
        - metrics: Dict with keys:
            - 'initial_error': Error before optimization
            - 'optimized_error': Error after optimization
            - 'improvement': Absolute improvement
            - 'improvement_percent': Percentage improvement
            - 'n_iter': Number of iterations run
            - 'converged': Whether optimization converged (error < initial_error)

    Raises:
        ValueError: If initial params cannot be extracted or optimization fails
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Parameter Optimization")
        print("=" * 80)

    # Step 1: Extract initial parameters
    if verbose:
        print("\n[Step 1/5] Extracting initial parameters...")

    try:
        initial_params = extract_initial_params_from_class(class_code)
    except Exception as e:
        raise ValueError(f"Failed to extract initial params: {str(e)}")

    if verbose:
        print(f"  Initial params: {initial_params[:5]}{'...' if len(initial_params) > 5 else ''}")

    # Step 2: Define objective function
    if verbose:
        print("\n[Step 2/5] Setting up objective function...")

    def objective_function(params_dict: Dict[str, float]) -> float:
        """
        Objective function to minimize: max_diff metric.

        Args:
            params_dict: Dictionary mapping 'p0', 'p1', ... to parameter values

        Returns:
            Error value (max_diff) to minimize
        """
        # Convert params dict to array
        params_array = np.array([params_dict[f'p{i}'] for i in range(len(initial_params))])

        try:
            # Convert class to JSON with these params
            ha_json = convert_python_class_to_json(class_code, params_array)

            # Simulate
            evaluator = HAEvaluator(
                ha_dict=ha_json,
                npz_file_path=npz_file_path,
                dt=dt,
                total_time=total_time
            )

            # Compute metrics
            evaluator.load_ground_truth()
            evaluator.simulate()
            metrics = evaluator.compute_metrics()

            # Return max_diff as error (primary metric)
            error = metrics.get('max_diff', float('inf'))

            # Penalize NaN/Inf
            if np.isnan(error) or np.isinf(error):
                return 1e10

            return error

        except Exception as e:
            # Simulation failure → high penalty
            if verbose:
                print(f"    Simulation error: {str(e)[:50]}")
            return 1e10

    # Step 3: Compute initial error
    if verbose:
        print("\n[Step 3/5] Computing initial error...")

    initial_params_dict = {f'p{i}': val for i, val in enumerate(initial_params)}
    initial_error = objective_function(initial_params_dict)

    if verbose:
        print(f"  Initial error: {initial_error:.6f}")

    # Step 4: Set up optimizer and search space
    if verbose:
        print(f"\n[Step 4/5] Running {optimizer_type} optimization ({n_iter} iterations)...")

    search_space = auto_scale_search_space(initial_params, scale_factor=10.0)

    if optimizer_type == 'simulated_annealing':
        optimizer = SimulatedAnnealingOptimizer(search_space)
    elif optimizer_type == 'hill_climbing':
        optimizer = HillClimbingOptimizer(search_space)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use 'simulated_annealing' or 'hill_climbing'.")

    # Run optimization
    try:
        optimizer.search(
            objective_function,
            n_iter=n_iter,
            initialize={'warm_start': [initial_params_dict]}  # Start from initial guess
        )
    except Exception as e:
        raise ValueError(f"Optimization failed: {str(e)}\n{traceback.format_exc()}")

    # Step 5: Extract results
    if verbose:
        print("\n[Step 5/5] Extracting optimized parameters...")

    # Get best parameters
    best_params_dict = optimizer.best_para
    optimized_params = np.array([best_params_dict[f'p{i}'] for i in range(len(initial_params))])
    optimized_error = optimizer.best_score

    # Compute metrics
    improvement = initial_error - optimized_error
    improvement_percent = (improvement / initial_error * 100) if initial_error > 0 else 0.0

    metrics = {
        'initial_error': initial_error,
        'optimized_error': optimized_error,
        'improvement': improvement,
        'improvement_percent': improvement_percent,
        'n_iter': n_iter,
        'converged': optimized_error < initial_error
    }

    if verbose:
        print(f"  Optimized error: {optimized_error:.6f}")
        print(f"  Improvement: {improvement:.6f} ({improvement_percent:.2f}%)")

    # Inject optimized params into class code
    optimized_class_code = inject_params_into_class(class_code, optimized_params)

    if verbose:
        print("\n" + "=" * 80)
        if metrics['converged']:
            print("✓ Optimization converged successfully")
        else:
            print("⚠️  Optimization did not improve error")
        print("=" * 80)

    return optimized_class_code, optimized_params, metrics


if __name__ == "__main__":
    # Test the optimizer with a simple example
    print("Testing ha_class_optimizer.py")
    print("=" * 80)

    # Example: Damped oscillator with noisy initial params
    # Ground truth: damping=-0.1, spring=-5.0, input_gain=1.0
    # Initial guess: damping=-0.5, spring=-10.0, input_gain=0.5 (wrong!)

    test_class = '''
class HybridAutomaton:
    """Test damped oscillator."""

    def __init__(self):
        # WRONG initial params (to test optimization)
        self.params = [
            -0.5,   # params[0]: damping (true: -0.1)
            -10.0,  # params[1]: spring (true: -5.0)
            0.5,    # params[2]: input gain (true: 1.0)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.var = "x1"
        self.input = "u1"
        self.dt = 0.001
        self.total_time = 10.0
        self.order = 2
        self.need_reset = False
        self.non_linear_items = ""

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u['u1']"
        raise ValueError(f"Unknown mode_id: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass

    def to_json(self):
        mode_eq = f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*u1"
        return {
            "automaton": {
                "var": self.var,
                "input": self.input,
                "mode": [{"id": 1, "eq": mode_eq}],
                "edge": []
            },
            "config": {
                "dt": self.dt,
                "total_time": self.total_time,
                "order": self.order,
                "need_reset": self.need_reset,
                "non_linear_items": self.non_linear_items
            }
        }
'''

    # Find a test NPZ file
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data_all', 'non_linear', 'duffing')
    if os.path.exists(test_data_dir):
        npz_files = [f for f in os.listdir(test_data_dir) if f.endswith('.npz')]
        if npz_files:
            test_npz_path = os.path.join(test_data_dir, npz_files[0])

            print(f"\nTest data: {test_npz_path}")
            print("\nRunning optimization (this may take 30-60 seconds)...")

            try:
                optimized_code, opt_params, metrics = optimize_ha_params(
                    class_code=test_class,
                    npz_file_path=test_npz_path,
                    n_iter=50,  # Reduced for testing
                    optimizer_type='simulated_annealing',
                    verbose=True
                )

                print("\n✓ Optimization completed successfully")
                print(f"\nOptimized params: {opt_params[:3]}")
                print(f"Metrics: {metrics}")

                if metrics['improvement_percent'] > 50:
                    print("\n✓ TEST PASSED: >50% error improvement achieved")
                else:
                    print(f"\n⚠️  Warning: Only {metrics['improvement_percent']:.1f}% improvement")

            except Exception as e:
                print(f"\n✗ Optimization failed: {e}")
                traceback.print_exc()
        else:
            print("\n⚠️  No NPZ files found in test data directory")
    else:
        print(f"\n⚠️  Test data directory not found: {test_data_dir}")
        print("Skipping full test. To test, provide a valid NPZ file path.")

    print("\n" + "=" * 80)
    print("Module test completed")
