"""
Pure Python class-based HA evaluation (no JSON conversion).

Directly executes and evaluates Python class code.
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime


def execute_python_class(class_code: str):
    """
    Execute Python class code and return the HybridAutomaton instance.

    Args:
        class_code: Python class definition string

    Returns:
        Instance of HybridAutomaton class

    Raises:
        Exception if execution fails
    """
    # Create isolated namespace
    namespace = {}

    # Execute the class definition
    try:
        exec(class_code, namespace)
    except SyntaxError as e:
        # Print problematic code section for debugging
        lines = class_code.split('\n')
        start = max(0, e.lineno - 3) if e.lineno else 0
        end = min(len(lines), (e.lineno + 2) if e.lineno else 10)
        print(f"\nSyntax error at line {e.lineno}:")
        for i in range(start, end):
            marker = " >>>" if i == e.lineno - 1 else "    "
            print(f"{marker} {i+1}: {lines[i]}")
        raise

    # Instantiate the class
    if 'HybridAutomaton' not in namespace:
        raise ValueError("Class code must define 'HybridAutomaton' class")

    ha_instance = namespace['HybridAutomaton']()

    return ha_instance


def evaluate_python_ha_class(
    class_code: str,
    input_data_path: str,
    output_dir: Optional[str] = None,
    iteration: int = 1,
    optimization_iters: int = 100,
    optimizer_type: str = "simulated_annealing"
) -> Tuple[bool, Dict, str, str, Optional[np.ndarray]]:
    """
    Evaluate Python class-based HA specification (pure Python, no JSON).

    Workflow:
    1. Execute class code → get HA instance
    2. Optimize parameters
    3. Simulate and compute metrics
    4. Generate feedback

    Args:
        class_code: Python class definition string
        input_data_path: Path to directory with .npz files
        output_dir: Where to save results
        iteration: Current iteration number
        optimization_iters: Number of optimization iterations
        optimizer_type: 'simulated_annealing' or 'hill_climbing'

    Returns:
        success: Whether evaluation succeeded
        metrics: Dict of metric values
        feedback: Formatted feedback string for next iteration
        optimized_class_code: Class code with optimized params
        optimized_params: NumPy array of optimized params
    """
    print("\n" + "=" * 80)
    print("PURE PYTHON CLASS EVALUATION")
    print("=" * 80)

    try:
        # Step 1: Execute class code
        print("\n[1/4] Executing Python class...")
        ha_instance = execute_python_class(class_code)
        print(f"✓ Class executed successfully")
        print(f"  - Variables: {ha_instance.var}")
        print(f"  - Inputs: {ha_instance.input}")
        print(f"  - Modes: {ha_instance.num_modes()}")
        print(f"  - Order: {ha_instance.order}")

    except Exception as e:
        error_msg = f"Failed to execute Python class: {e}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, {}, error_msg, class_code, None

    # Find NPZ file
    test_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_files:
        return False, {}, f"No .npz files found in {input_data_path}", class_code, None

    npz_path = os.path.join(input_data_path, test_files[0])

    # Step 2: Optimize parameters
    print(f"\n[2/4] Optimizing parameters ({optimization_iters} iterations)...")
    try:
        from utils.ha_class_optimizer import optimize_ha_params

        optimized_class_code, optimized_params, opt_metrics = optimize_ha_params(
            class_code=class_code,
            npz_file_path=npz_path,
            n_iter=optimization_iters,
            optimizer_type=optimizer_type
        )

        print(f"✓ Optimization complete")
        init_err = opt_metrics.get('initial_error', 'N/A')
        final_err = opt_metrics.get('final_error', 'N/A')
        improvement = opt_metrics.get('improvement_percentage', 0)
        print(f"  - Initial error: {init_err:.6f}" if isinstance(init_err, (int, float)) else f"  - Initial error: {init_err}")
        print(f"  - Final error: {final_err:.6f}" if isinstance(final_err, (int, float)) else f"  - Final error: {final_err}")
        print(f"  - Improvement: {improvement:.2f}%" if isinstance(improvement, (int, float)) else f"  - Improvement: {improvement}")

        # Re-execute with optimized params
        ha_instance = execute_python_class(optimized_class_code)

    except Exception as e:
        print(f"⚠ Optimization failed: {e}. Using initial params.")
        optimized_class_code = class_code
        optimized_params = np.array(ha_instance.params[:10])
        opt_metrics = {}

    # Step 3: Simulate and evaluate
    print(f"\n[3/4] Simulating with optimized parameters...")
    try:
        from utils.ha_class_simulator import PythonClassHAEvaluator

        evaluator = PythonClassHAEvaluator(
            ha_instance=ha_instance,
            npz_file_path=npz_path,
            dt=getattr(ha_instance, 'dt', 0.001),
            total_time=getattr(ha_instance, 'total_time', 10.0)
        )

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join("evaluation_results", f"iter_{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        # Run evaluation with plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'ha_eval_{timestamp}.png')

        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path=plot_path,
            print_metrics=True,
            show_plot=False
        )

        metrics = evaluator.metrics
        print(f"✓ Simulation complete")

    except Exception as e:
        error_msg = f"Simulation failed: {e}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return False, {}, error_msg, optimized_class_code, optimized_params

    # Step 4: Generate feedback
    print(f"\n[4/4] Generating feedback...")

    # Format optimization metrics
    init_err = opt_metrics.get('initial_error', 'N/A')
    final_err = opt_metrics.get('final_error', 'N/A')
    improvement = opt_metrics.get('improvement_percentage', 'N/A')

    init_err_str = f"{init_err:.6f}" if isinstance(init_err, (int, float)) else str(init_err)
    final_err_str = f"{final_err:.6f}" if isinstance(final_err, (int, float)) else str(final_err)
    improvement_str = f"{improvement:.2f}%" if isinstance(improvement, (int, float)) else str(improvement)

    feedback = f"""## Iteration {iteration} Results

### Python Class (Optimized)
```python
{optimized_class_code}
```

### Optimized Parameters
{np.array2string(optimized_params, precision=4, separator=', ')}

### Optimization Metrics
- Initial error: {init_err_str}
- Final error: {final_err_str}
- Improvement: {improvement_str}

### Evaluation Metrics
{metrics_text}

### Analysis
"""

    # Add specific advice based on metrics
    max_diff = metrics.get('max_diff', float('inf'))
    tc = metrics.get('tc', float('inf'))

    if max_diff > 0.1:
        feedback += "- **State error is high**: Check if dynamics equations are correct\n"
    elif max_diff > 0.01:
        feedback += "- **State error moderate**: Fine-tune parameters or add nonlinear terms\n"
    else:
        feedback += "- **State error good**: Dynamics are accurate\n"

    if tc > 0.1:
        feedback += "- **Mode switching error**: Review guard conditions and thresholds\n"
    elif tc > 0.01:
        feedback += "- **Mode switching acceptable**: Consider fine-tuning thresholds\n"
    else:
        feedback += "- **Mode switching excellent**: Transitions are accurate\n"

    # Save full results
    results_file = os.path.join(output_dir, f'results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(feedback)
        f.write(f"\n\nPlot saved to: {plot_path}\n")

    print(f"✓ Feedback generated")
    print(f"\nResults saved to: {output_dir}")

    return True, metrics, feedback, optimized_class_code, optimized_params


# Test
if __name__ == "__main__":
    # Test with simple template
    from utils.ha_class_template import get_simple_ha_template

    test_class = get_simple_ha_template(num_variables=1, num_inputs=1)

    print("Testing evaluation with simple template:")
    print("="*80)

    success, metrics, feedback, opt_class, opt_params = evaluate_python_ha_class(
        class_code=test_class,
        input_data_path="data_all/non_linear/duffing",
        output_dir="test_eval",
        iteration=0,
        optimization_iters=10  # Small for testing
    )

    if success:
        print("\n✓ Evaluation succeeded!")
        print(f"Metrics: {metrics}")
    else:
        print(f"\n✗ Evaluation failed: {feedback}")
