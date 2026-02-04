"""
Hybrid Automaton Parameter Optimization using Nevergrad

This script demonstrates how to use gradient-free optimization (Nevergrad)
to automatically tune parameters in a Hybrid Automaton specification.

Example: Optimizing a bouncing ball system's gravity and restitution coefficient.
"""

import nevergrad as ng
import numpy as np
import sys
import os
import copy

# Add path for HA evaluation imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils/Dainarx_code'))
from HA_evaluation import HAEvaluator


# ============================================================================
# Hybrid Automaton Template
# ============================================================================
# This is the bouncing ball system with parametrized gravity (g) and
# restitution coefficient (r). We'll optimize these parameters to match
# ground truth data.

def create_ha_spec(gravity: float, restitution: float) -> dict:
    """
    Create a Hybrid Automaton specification for a bouncing ball.

    Args:
        gravity: Gravitational acceleration (positive value, will be negated)
        restitution: Coefficient of restitution (0 < r < 1)

    Returns:
        Dictionary containing the HA specification
    """
    return {
        "automaton": {
            "var": "x1, x2",  # x1 = position, x2 = velocity
            "mode": [
                {
                    "id": 1,
                    # ODE: dx1/dt = x2, dx2/dt = -gravity
                    "eq": f"x1[1] = x2[0], x2[1] = -{gravity}"
                }
            ],
            "edge": [
                {
                    "direction": "1 -> 1",  # Self-loop (bouncing)
                    "condition": "x1 <= 0",  # Ball hits ground
                    "reset": {
                        "x1": [0],  # Position resets to 0
                        "x2": [f"-{restitution} * x2[0]"]  # Velocity reverses with loss
                    }
                }
            ]
        },
        "config": {
            "dt": 0.01,
            "total_time": 10.0,
            "order": 1,
            "self_loop": True
        }
    }


# ============================================================================
# Objective Function
# ============================================================================

def objective_function(params: np.ndarray, npz_file_path: str, verbose: bool = False) -> float:
    """
    Objective function for optimization: returns trajectory error.

    Args:
        params: Array of [gravity, restitution]
        npz_file_path: Path to ground truth NPZ file
        verbose: Whether to print intermediate results

    Returns:
        Mean trajectory difference (lower is better)
    """
    gravity, restitution = params

    # Ensure restitution is in valid range (0, 1)
    restitution = np.clip(restitution, 0.01, 0.99)
    gravity = np.clip(gravity, 0.1, 20.0)

    # Create HA specification with current parameters
    ha_spec = create_ha_spec(gravity, restitution)

    try:
        # Create evaluator and compute metrics
        evaluator = HAEvaluator(
            ha_dict=ha_spec,
            npz_file_path=npz_file_path,
            dt=0.01,
            total_time=10.0
        )

        # Run simulation and compute metrics
        evaluator.load_ground_truth()
        evaluator.simulate()
        metrics = evaluator.compute_metrics()

        # Use mean_diff as the objective (trajectory error)
        mean_diff = metrics.get('mean_diff', float('inf'))

        if mean_diff is None:
            mean_diff = float('inf')

        if verbose:
            print(f"  g={gravity:.4f}, r={restitution:.4f} -> mean_diff={mean_diff:.6f}")

        return mean_diff

    except Exception as e:
        if verbose:
            print(f"  g={gravity:.4f}, r={restitution:.4f} -> ERROR: {e}")
        return float('inf')  # Return large value on error


# ============================================================================
# Main Optimization
# ============================================================================

def optimize_ha_parameters(
    npz_file_path: str,
    budget: int = 100,
    verbose: bool = True
) -> dict:
    """
    Optimize Hybrid Automaton parameters using Nevergrad.

    Args:
        npz_file_path: Path to ground truth trajectory data
        budget: Number of optimization iterations
        verbose: Whether to print progress

    Returns:
        Dictionary with optimal parameters and final error
    """

    # Define parametrization with bounds
    # ng.p.Scalar creates a continuous parameter
    parametrization = ng.p.Instrumentation(
        # Gravity: typical range 5-15 m/s², starting guess around 10
        gravity=ng.p.Scalar(lower=5.0, upper=15.0).set_integer_casting(),
        # Restitution: coefficient between 0.5 and 1.0 (most balls)
        restitution=ng.p.Scalar(lower=0.5, upper=0.99)
    )

    # Create optimizer
    # NGOpt automatically selects a good algorithm based on budget and dimensionality
    optimizer = ng.optimizers.NGOpt(
        parametrization=parametrization,
        budget=budget,
        num_workers=1  # Sequential evaluation
    )

    if verbose:
        print(f"Starting optimization with budget={budget}")
        print(f"Ground truth file: {npz_file_path}")
        print("-" * 60)

    # Define the objective wrapper that Nevergrad will call
    def nevergrad_objective(*args, **kwargs):
        gravity = kwargs.get('gravity', args[0] if args else 9.8)
        restitution = kwargs.get('restitution', args[1] if len(args) > 1 else 0.9)
        params = np.array([gravity, restitution])
        return objective_function(params, npz_file_path, verbose=verbose)

    # Run optimization
    recommendation = optimizer.minimize(nevergrad_objective)

    # Extract optimal parameters
    optimal_gravity = recommendation.kwargs['gravity']
    optimal_restitution = recommendation.kwargs['restitution']
    optimal_error = recommendation.loss

    if verbose:
        print("-" * 60)
        print("Optimization complete!")
        print(f"Optimal gravity: {optimal_gravity:.4f} m/s²")
        print(f"Optimal restitution: {optimal_restitution:.4f}")
        print(f"Final mean_diff error: {optimal_error:.6f}")

    return {
        'gravity': optimal_gravity,
        'restitution': optimal_restitution,
        'error': optimal_error,
        'ha_spec': create_ha_spec(optimal_gravity, optimal_restitution)
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Path to ground truth data
    npz_file = "data_all/ATVA/ball_g/ground_truth_0.npz"

    # Check if file exists
    if not os.path.exists(npz_file):
        print(f"Error: Ground truth file not found: {npz_file}")
        print("Please ensure the data_all/ATVA/ball_g/ directory contains ground truth files.")
        sys.exit(1)

    print("=" * 60)
    print("Hybrid Automaton Parameter Optimization Demo")
    print("System: Bouncing Ball")
    print("Parameters to optimize: gravity (g), restitution (r)")
    print("=" * 60)
    print()

    # Known ground truth values for bouncing ball:
    # gravity = 9.8 m/s², restitution = 0.9
    print("Ground truth parameters (for reference):")
    print("  gravity = 9.8 m/s²")
    print("  restitution = 0.9")
    print()

    # Run optimization with a small budget for demo
    result = optimize_ha_parameters(
        npz_file_path=npz_file,
        budget=50,  # Increase for better results
        verbose=True
    )

    print()
    print("=" * 60)
    print("Final optimized HA specification:")
    print("=" * 60)
    import json
    print(json.dumps(result['ha_spec'], indent=2))
