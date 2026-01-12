import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os


@dataclass
class HAHyperparameters:
    """Stores hyperparameters for HA learning experiments."""
    # Manager agent configuration
    manager_model: str = ""
    manager_type: str = "CodeAgent"

    # Managed agents configuration
    managed_agents_count: int = 0
    managed_agents_list: List[str] = field(default_factory=list)
    managed_agents_model: str = ""

    # Tools configuration
    tools_list: List[str] = field(default_factory=list)
    image_tool_model: str = ""
    review_tool_model: str = ""
    summarize_tool_model: str = ""

    # Iteration and feedback configuration
    max_iterations: int = 3
    feedback_top_k: int = 3
    feedback_min_gap: float = 0.005
    target_error: float = 0.01
    no_improvement_patience: int = 3

    # Data configuration
    input_data_path: str = ""
    system_name: str = ""
    num_variables: int = 0
    num_inputs: int = 0

    # Other configuration
    use_json_schema: bool = True

    def to_string(self) -> str:
        """Convert hyperparameters to a formatted string for logging."""
        lines = [
            "Experiment Hyperparameters",
            "=" * 60,
            "",
            "--- Manager Agent ---",
            f"Model: {self.manager_model}",
            f"Type: {self.manager_type}",
            "",
            "--- Managed Agents ---",
            f"Count: {self.managed_agents_count}",
            f"Agent Names: {', '.join(self.managed_agents_list) if self.managed_agents_list else 'None'}",
            f"Model: {self.managed_agents_model if self.managed_agents_model else 'N/A'}",
            "",
            "--- Tools ---",
            f"Tools List: {', '.join(self.tools_list) if self.tools_list else 'None'}",
            f"Image Tool Model: {self.image_tool_model if self.image_tool_model else 'N/A'}",
            f"Review Tool Model: {self.review_tool_model if self.review_tool_model else 'N/A'}",
            f"Summarize Tool Model: {self.summarize_tool_model if self.summarize_tool_model else 'N/A'}",
            "",
            "--- Iteration Configuration ---",
            f"Max Iterations: {self.max_iterations}",
            f"Feedback Top-K: {self.feedback_top_k}",
            f"Feedback Min Gap: {self.feedback_min_gap}",
            f"Target Error: {self.target_error}",
            f"No Improvement Patience: {self.no_improvement_patience}",
            "",
            "--- Data Configuration ---",
            f"Input Data Path: {self.input_data_path}",
            f"System Name: {self.system_name}",
            f"Num Variables: {self.num_variables}",
            f"Num Inputs: {self.num_inputs}",
            "",
            "--- Other ---",
            f"Use JSON Schema: {self.use_json_schema}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert hyperparameters to a dictionary."""
        return {
            "manager_model": self.manager_model,
            "manager_type": self.manager_type,
            "managed_agents_count": self.managed_agents_count,
            "managed_agents_list": self.managed_agents_list,
            "managed_agents_model": self.managed_agents_model,
            "tools_list": self.tools_list,
            "image_tool_model": self.image_tool_model,
            "review_tool_model": self.review_tool_model,
            "summarize_tool_model": self.summarize_tool_model,
            "max_iterations": self.max_iterations,
            "feedback_top_k": self.feedback_top_k,
            "feedback_min_gap": self.feedback_min_gap,
            "target_error": self.target_error,
            "no_improvement_patience": self.no_improvement_patience,
            "input_data_path": self.input_data_path,
            "system_name": self.system_name,
            "num_variables": self.num_variables,
            "num_inputs": self.num_inputs,
            "use_json_schema": self.use_json_schema,
        }


@dataclass
class IterationResult:
    """Tracks results from a single iteration of the HA-Scientist loop."""
    iteration: int
    ha_specification: Optional[Dict] = None
    metrics: Dict = field(default_factory=dict)
    feedback: str = ""
    success: bool = False
    error_value: float = float('inf')
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    # New fields for Python class-based format
    class_code: Optional[str] = None  # Python class code (if class format used)
    optimized_params: Optional[np.ndarray] = None  # Optimized parameters (if optimization applied)


class ResultsAggregator:
    """
    Aggregates results across iterations for intelligent feedback selection.
    Inspired by reference_code.py's _create_previous_turn_context pattern.
    """
    def __init__(self, top_k: int = 3, min_gap: float = 0.005):
        """
        Initialize the results aggregator.

        Args:
            top_k: Maximum number of distinct results to include in feedback
            min_gap: Minimum error gap between selected results for diversity
        """
        self.results: List[IterationResult] = []
        self.top_k = top_k
        self.min_gap = min_gap
        self.best_result: Optional[IterationResult] = None
        self.best_error: float = float('inf')

    def add_result(self, result: IterationResult) -> None:
        """Add a new iteration result and update best tracking."""
        self.results.append(result)

        if result.success and result.error_value < self.best_error:
            self.best_error = result.error_value
            self.best_result = result
            print(f"  [Aggregator] New best result! Error: {self.best_error:.6f}")

    def get_top_k_feedback(self) -> str:
        """
        Generate feedback context from top-k diverse, best-performing specifications.

        Uses minimum gap filtering to ensure diverse examples (from reference_code.py pattern).

        Returns:
            Formatted feedback string with sorted, distinct results
        """
        # Filter successful results with valid error values
        valid_results = [
            r for r in self.results
            if r.success and r.error_value is not None and r.error_value < float('inf')
        ]

        if not valid_results:
            return ""

        # Sort by error value (ascending - best first)
        sorted_results = sorted(valid_results, key=lambda x: x.error_value)

        # Select distinct results with minimum gap (diversity filtering)
        distinct_results: List[IterationResult] = []
        last_accepted_error = float('-inf')

        for result in sorted_results:
            if result.error_value - last_accepted_error >= self.min_gap:
                distinct_results.append(result)
                last_accepted_error = result.error_value
                if len(distinct_results) >= self.top_k:
                    break

        if not distinct_results:
            return ""

        # Build structured feedback context
        context_lines = [
            "\n\n## Previously Explored HA Specifications",
            "The following specifications have been explored in previous iterations.",
            "Performance is ranked from best (lowest error) to worst. Use these as inspiration.",
            "\n--- Explored Specifications (Ranked) ---"
        ]

        for i, result in enumerate(distinct_results, 1):
            context_lines.append(f"\n### Rank {i} (Iteration {result.iteration})")
            context_lines.append(f"**Error**: {result.error_value:.6f}")

            # Show Python class code if available, otherwise JSON
            if result.class_code:
                # Python class format
                class_code_str = result.class_code if len(result.class_code) <= 1500 else result.class_code[:1500] + "\n... (truncated)"
                context_lines.append(f"**Format**: Python class")
                context_lines.append(f"```python\n{class_code_str}\n```")

                # Show optimized params if available
                if result.optimized_params is not None:
                    params_str = str(result.optimized_params.tolist()[:5]) + "..." if len(result.optimized_params) > 5 else str(result.optimized_params.tolist())
                    context_lines.append(f"**Optimized Params**: {params_str}")
            else:
                # JSON format
                ha_spec_str = json.dumps(result.ha_specification, indent=2) if result.ha_specification else "N/A"
                # Truncate if too long
                if len(ha_spec_str) > 1500:
                    ha_spec_str = ha_spec_str[:1500] + "\n... (truncated)"
                context_lines.append(f"**Format**: JSON")
                context_lines.append(f"```json\n{ha_spec_str}\n```")

            # Include key metrics if available
            if result.metrics:
                metrics_summary = ", ".join([
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in result.metrics.items()
                    if k in ['mean_diff', 'max_diff', 'tc', 'rmse']
                ])
                if metrics_summary:
                    context_lines.append(f"**Metrics**: {metrics_summary}")

        context_lines.append("\n-----------------------------------------\n")
        return "\n".join(context_lines)

    def get_latest_feedback(self) -> str:
        """Get feedback from the most recent iteration only."""
        if not self.results:
            return ""
        return self.results[-1].feedback

    def should_early_stop(self,
                          target_error: float = 0.01,
                          min_iterations: int = 2,
                          no_improvement_patience: int = 3) -> Tuple[bool, str]:
        """
        Determine if early stopping criteria are met.

        Inspired by reference_code.py's early termination logic.

        Args:
            target_error: Stop if best error falls below this threshold
            min_iterations: Minimum iterations before allowing early stop
            no_improvement_patience: Stop if no improvement for this many iterations

        Returns:
            Tuple of (should_stop, reason)
        """
        if len(self.results) < min_iterations:
            return False, ""

        # Check target achieved
        if self.best_error < target_error:
            return True, f"Target error achieved: {self.best_error:.6f} < {target_error}"

        # Check for very low error (near-perfect fit)
        if self.best_error < 0.0001:
            return True, f"Near-perfect fit achieved: {self.best_error:.6e}"

        # Check for no improvement patience
        if len(self.results) >= no_improvement_patience:
            recent_errors = [r.error_value for r in self.results[-no_improvement_patience:]]
            if all(e >= self.best_error for e in recent_errors):
                # Check if we've plateaued
                error_range = max(recent_errors) - min(recent_errors)
                if error_range < 0.001:  # Less than 0.1% variation
                    return True, f"No improvement for {no_improvement_patience} iterations"

        return False, ""

    def get_dynamic_error_threshold(self) -> float:
        """
        Calculate dynamic error threshold based on current best.

        Inspired by reference_code.py's adaptive MAPE target adjustment.
        """
        if self.best_error >= float('inf') or self.best_error <= 0:
            return 0.1  # Default threshold

        # Set next target to one order of magnitude below current best
        next_target = 10 ** np.floor(np.log10(self.best_error))
        if next_target >= self.best_error:
            next_target /= 10.0

        return max(next_target, 1e-8)  # Floor at 1e-8

def evaluate_ha_specification(agent_result, input_data_path: str, output_dir: str = None) -> bool:
    """
    Evaluate the generated Hybrid Automaton specification against ground truth data.

    Args:
        agent_result: Result from the agent.run() call (can be dict or str)
        input_data_path: Path to the directory containing test .npz files
        output_dir: Directory to save evaluation results (default: './evaluation_results')

    Returns:
        bool: True if evaluation succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Testing the generated Hybrid Automaton specification")
    print("=" * 80)

    # Import HA evaluation module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'Dainarx_code'))
    from HA_evaluation import HAEvaluator
    
    # Import HA specification validator
    from utils.ha_spec_validator import preprocess_ha_for_evaluation

    # Use the validator to extract and fix HA specification
    print("\n--- HA Specification Validation ---")
    ha_specification, is_valid, validation_message = preprocess_ha_for_evaluation(agent_result)
    print(validation_message)
    
    if not is_valid:
        print("\nError: HA specification validation failed")
        if ha_specification is not None:
            print(f"Partially extracted spec: {json.dumps(ha_specification, indent=2)[:500]}...")
        return False

    # Final structure check (should pass after validation, but double-check)
    if ha_specification is None or 'automaton' not in ha_specification or 'config' not in ha_specification:
        print("\nWarning: Could not extract valid HA specification from agent output for evaluation")
        print("Expected dictionary with 'automaton' and 'config' keys")
        return False

    print("\n--- HA Specification Summary ---")
    print(f"Variables: {ha_specification['automaton'].get('var', 'N/A')}")
    print(f"Inputs: {ha_specification['automaton'].get('input', 'N/A')}")
    print(f"Modes: {len(ha_specification['automaton'].get('mode', []))}")
    print(f"Edges: {len(ha_specification['automaton'].get('edge', []))}")

    # Find test data file in the input data path
    test_data_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not test_data_files:
        print(f"\nWarning: No .npz test data files found in {input_data_path}")
        return False

    # Use first test file found
    npz_file_path = os.path.join(input_data_path, test_data_files[0])
    print(f"\nUsing test data file: {npz_file_path}")

    # Load ground truth data to check dimensions
    gt_data = np.load(npz_file_path, allow_pickle=True)
    gt_num_vars = gt_data['state'].shape[0]
    
    # Count variables in HA spec
    var_str = ha_specification['automaton'].get('var', '')
    ha_num_vars = len([v.strip() for v in var_str.split(',') if v.strip()])
    
    print(f"\n--- Dimension Check ---")
    print(f"Ground truth state variables: {gt_num_vars}")
    print(f"HA specification variables: {ha_num_vars} ({var_str})")
    
    if ha_num_vars != gt_num_vars:
        print(f"\n⚠️ ERROR: Variable count mismatch!")
        print(f"HA spec defines {ha_num_vars} variable(s), but ground truth has {gt_num_vars} variable(s).")
        print(f"The LLM may have incorrectly converted to state-space form.")
        print(f"For a {gt_num_vars}-variable system, use higher-order ODEs (e.g., x1[2] = ...) instead of multiple 1st-order ODEs.")
        return False

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create evaluator
        evaluator = HAEvaluator(
            ha_dict=ha_specification,
            npz_file_path=npz_file_path,
            dt=ha_specification['config'].get('dt', 0.001),
            total_time=ha_specification['config'].get('total_time', 10.0)
        )

        # Run evaluation with overlay plot
        save_path = os.path.join(output_dir, 'ha_evaluation_comparison.png')

        print("\nRunning HA evaluation...")
        metrics_text, _ = evaluator(
            plot_mode='overlay',
            save_path=save_path,
            print_metrics=True
        )

        print(f"\nEvaluation complete! Results saved to: {save_path}")

        # Save metrics and HA specification to file
        metrics_file = os.path.join(output_dir, 'ha_evaluation_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("Hybrid Automaton Evaluation Metrics\n")
            f.write("=" * 80 + "\n\n")
            f.write(metrics_text)
            f.write("\n\nHA Specification:\n")
            f.write(json.dumps(ha_specification, indent=2))
        print(f"Metrics saved to: {metrics_file}")

        return True

    except Exception as e:
        print(f"\nError during HA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False