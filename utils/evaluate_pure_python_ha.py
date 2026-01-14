"""Pure Python class-based HA evaluation (no JSON conversion).

Supports classes that inherit from HybridAutomatonBase.
"""

import os
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime

from utils.ha_base_class import HybridAutomatonBase


def execute_python_class(class_code: str):
    """
    Execute Python class code and return the HybridAutomaton instance.

    The execution namespace includes HybridAutomatonBase for inheritance support.
    """
    # Create namespace with base class available for import
    namespace = {
        'HybridAutomatonBase': HybridAutomatonBase,
    }

    # Also simulate the import statement by adding to namespace
    # This handles code that starts with: from utils.ha_base_class import HybridAutomatonBase
    class FakeModule:
        HybridAutomatonBase = HybridAutomatonBase

    namespace['utils'] = type('utils', (), {'ha_base_class': FakeModule})()

    try:
        # Remove the import statement if present (we've already provided the class)
        code_lines = class_code.split('\n')
        filtered_lines = []
        for line in code_lines:
            # Skip import lines for ha_base_class
            if 'from utils.ha_base_class import' in line:
                continue
            if 'import utils.ha_base_class' in line:
                continue
            filtered_lines.append(line)
        filtered_code = '\n'.join(filtered_lines)

        exec(filtered_code, namespace)
    except SyntaxError as e:
        lines = class_code.split('\n')
        start, end = max(0, (e.lineno or 1) - 3), min(len(lines), (e.lineno or 1) + 2)
        print(f"\nSyntax error at line {e.lineno}:")
        for i in range(start, end):
            print(f"{' >>>' if i == e.lineno - 1 else '    '} {i+1}: {lines[i]}")
        raise

    # Look for HybridAutomaton class (or any class inheriting from HybridAutomatonBase)
    ha_class = None

    # First, try to find 'HybridAutomaton' explicitly
    if 'HybridAutomaton' in namespace:
        ha_class = namespace['HybridAutomaton']
    else:
        # Look for any class that inherits from HybridAutomatonBase
        for name, obj in namespace.items():
            if (isinstance(obj, type) and
                issubclass(obj, HybridAutomatonBase) and
                obj is not HybridAutomatonBase):
                ha_class = obj
                break

    if ha_class is None:
        raise ValueError(
            "Class code must define 'HybridAutomaton' class or a class inheriting from HybridAutomatonBase"
        )

    return ha_class()


def _fmt(val, fmt=".6f"):
    """Format numeric value or return as string."""
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)


def _get_analysis(metrics: Dict) -> str:
    """Generate analysis text based on metrics."""
    max_diff = metrics.get('max_diff', float('inf'))
    tc = metrics.get('tc', float('inf'))

    state_msg = ("high - check dynamics" if max_diff > 0.1
                 else "moderate - tune parameters" if max_diff > 0.01
                 else "good")
    mode_msg = ("error - review guards" if tc > 0.1
                else "acceptable" if tc > 0.01
                else "excellent")

    return f"- State error: {state_msg}\n- Mode switching: {mode_msg}"


def evaluate_python_ha_class(
    class_code: str,
    input_data_path: str,
    output_dir: Optional[str] = None,
    iteration: int = 1,
    optimization_iters: int = 100,
    optimizer_type: str = "simulated_annealing"
) -> Tuple[bool, Dict, str, str, Optional[np.ndarray], Optional[str]]:
    """
    Evaluate Python class-based HA specification.

    Returns: (success, metrics, feedback, optimized_class_code, optimized_params, plot_path)
    """
    print(f"\n{'='*60}\nEVALUATION (iter {iteration})\n{'='*60}")

    # Step 1: Execute class
    try:
        ha_instance = execute_python_class(class_code)
        print(f"Class: var={ha_instance.var}, modes={ha_instance.num_modes()}")

        # Check if it properly inherits from base class
        if isinstance(ha_instance, HybridAutomatonBase):
            print("  (Inherits from HybridAutomatonBase)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, {}, f"Execution failed: {e}", class_code, None, None

    # Find NPZ file
    test_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_files:
        return False, {}, f"No .npz files in {input_data_path}", class_code, None, None
    npz_path = os.path.join(input_data_path, test_files[0])

    # Step 2: Optimize
    print(f"Optimizing ({optimization_iters} iters)...")
    try:
        from utils.ha_class_optimizer import optimize_ha_params
        optimized_class_code, optimized_params, opt_metrics = optimize_ha_params(
            class_code=class_code, npz_file_path=npz_path,
            n_iter=optimization_iters, optimizer_type=optimizer_type
        )
        print(f"Optimization: {_fmt(opt_metrics.get('initial_error', 'N/A'))} -> "
              f"{_fmt(opt_metrics.get('final_error', 'N/A'))}")
        ha_instance = execute_python_class(optimized_class_code)
    except Exception as e:
        print(f"Optimization failed: {e}")
        optimized_class_code, optimized_params, opt_metrics = class_code, np.array(ha_instance.params[:10]), {}

    # Step 3: Simulate
    print("Simulating...")
    try:
        from utils.ha_class_simulator import PythonClassHAEvaluator
        evaluator = PythonClassHAEvaluator(
            ha_instance=ha_instance, npz_file_path=npz_path,
            dt=getattr(ha_instance, 'dt', 0.001),
            total_time=getattr(ha_instance, 'total_time', 10.0)
        )

        output_dir = output_dir or os.path.join("evaluation_results", f"iter_{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'ha_eval_{timestamp}.png')
        metrics_text, _ = evaluator(plot_mode='overlay', save_path=plot_path,
                                    print_metrics=True, show_plot=False)
        metrics = evaluator.metrics
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, {}, f"Simulation failed: {e}", optimized_class_code, optimized_params, None

    # Step 4: Generate feedback
    feedback = f"""## Iteration {iteration} Results

### Optimized Class
```python
{optimized_class_code}
```

### Parameters
{np.array2string(optimized_params, precision=4, separator=', ')}

### Optimization: {_fmt(opt_metrics.get('initial_error', 'N/A'))} -> {_fmt(opt_metrics.get('final_error', 'N/A'))} ({_fmt(opt_metrics.get('improvement_percentage', 0), '.1f')}%)

### Metrics
{metrics_text}

### Analysis
{_get_analysis(metrics)}
"""

    with open(os.path.join(output_dir, f'results_{timestamp}.txt'), 'w') as f:
        f.write(feedback)

    print(f"Results saved to: {output_dir}")
    return True, metrics, feedback, optimized_class_code, optimized_params, plot_path


def get_execution_namespace():
    """
    Get the namespace dict for executing HA class code.

    Use this when you need to provide the execution context externally.
    """
    class FakeModule:
        HybridAutomatonBase = HybridAutomatonBase

    return {
        'HybridAutomatonBase': HybridAutomatonBase,
        'utils': type('utils', (), {'ha_base_class': FakeModule})(),
    }


# Test
if __name__ == "__main__":
    # Test with simple template
    from utils.ha_class_template import get_simple_ha_template

    test_class = get_simple_ha_template(num_variables=1, num_inputs=1)

    print("Testing evaluation with simple template:")
    print("="*80)
    print(test_class)
    print("="*80)

    success, metrics, feedback, opt_class, opt_params, plot_path = evaluate_python_ha_class(
        class_code=test_class,
        input_data_path="data_all/non_linear/duffing",
        output_dir="test_eval",
        iteration=0,
        optimization_iters=10  # Small for testing
    )

    if success:
        print("\n Evaluation succeeded!")
        print(f"Metrics: {metrics}")
        print(f"Plot saved to: {plot_path}")
    else:
        print(f"\n Evaluation failed: {feedback}")
