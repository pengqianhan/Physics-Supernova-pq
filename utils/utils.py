import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import sys
import os


@dataclass
class HAHyperparameters:
    """Stores hyperparameters for HA learning experiments."""
    manager_model: str = ""
    manager_type: str = "CodeAgent"
    managed_agents_count: int = 0
    managed_agents_list: List[str] = field(default_factory=list)
    managed_agents_model: str = ""
    tools_list: List[str] = field(default_factory=list)
    image_tool_model: str = ""
    review_tool_model: str = ""
    summarize_tool_model: str = ""
    max_iterations: int = 3
    feedback_top_k: int = 3
    feedback_min_gap: float = 0.005
    target_error: float = 0.01
    no_improvement_patience: int = 3
    input_data_path: str = ""
    system_name: str = ""
    num_variables: int = 0
    num_inputs: int = 0
    use_json_schema: bool = True

    def to_string(self) -> str:
        """Convert hyperparameters to a formatted string for logging."""
        sections = [
            ("Manager Agent", [("Model", self.manager_model), ("Type", self.manager_type)]),
            ("Managed Agents", [
                ("Count", self.managed_agents_count),
                ("Names", ', '.join(self.managed_agents_list) or 'None'),
                ("Model", self.managed_agents_model or 'N/A')
            ]),
            ("Tools", [
                ("List", ', '.join(self.tools_list) or 'None'),
                ("Image Model", self.image_tool_model or 'N/A'),
                ("Review Model", self.review_tool_model or 'N/A'),
                ("Summarize Model", self.summarize_tool_model or 'N/A')
            ]),
            ("Iteration", [
                ("Max", self.max_iterations), ("Top-K", self.feedback_top_k),
                ("Min Gap", self.feedback_min_gap), ("Target Error", self.target_error),
                ("Patience", self.no_improvement_patience)
            ]),
            ("Data", [
                ("Path", self.input_data_path), ("System", self.system_name),
                ("Variables", self.num_variables), ("Inputs", self.num_inputs)
            ])
        ]
        lines = ["Experiment Hyperparameters", "=" * 60]
        for title, items in sections:
            lines.extend(["", f"--- {title} ---"])
            lines.extend(f"{k}: {v}" for k, v in items)
        lines.extend(["", f"Use JSON Schema: {self.use_json_schema}", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


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
    # Fields for Python class-based format
    class_code: Optional[str] = None  # Python class code (if class format used)
    optimized_params: Optional[np.ndarray] = None  # Optimized parameters (if optimization applied)
    # Structured output fields
    analysis_process: Optional[str] = None  # ~200 word analysis from agent
    llm_critique: Optional[str] = None  # LLM-generated critique based on metrics


class ResultsAggregator:
    """Aggregates results across iterations for intelligent feedback selection."""

    def __init__(self, top_k: int = 3, min_gap: float = 0.005):
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

    def _select_diverse_results(self) -> List[IterationResult]:
        """Select top-k diverse results with minimum gap filtering."""
        valid = [r for r in self.results
                 if r.success and r.error_value is not None and r.error_value < float('inf')]
        if not valid:
            return []

        sorted_results = sorted(valid, key=lambda x: x.error_value)
        selected = []
        last_error = float('-inf')

        for r in sorted_results:
            if r.error_value - last_error >= self.min_gap:
                selected.append(r)
                last_error = r.error_value
                if len(selected) >= self.top_k:
                    break
        return selected

    def get_top_k_results(self) -> List[IterationResult]:
        """Get top-k diverse results as IterationResult objects."""
        return self._select_diverse_results()

    def get_top_k_feedback(self) -> str:
        """Generate feedback context from top-k diverse specifications."""
        results = self._select_diverse_results()
        if not results:
            return ""

        lines = [
            "\n\n## Previously Explored HA Specifications",
            "Performance ranked from best to worst.\n"
        ]

        for i, r in enumerate(results, 1):
            lines.append(f"### Rank {i} (Iteration {r.iteration}) - Error: {r.error_value:.6f}")

            if r.class_code:
                code = r.class_code[:1500] + "\n..." if len(r.class_code) > 1500 else r.class_code
                lines.append(f"```python\n{code}\n```")
                if r.optimized_params is not None:
                    params = r.optimized_params.tolist()
                    lines.append(f"Params: {params[:5]}{'...' if len(params) > 5 else ''}")
            elif r.ha_specification:
                spec = json.dumps(r.ha_specification, indent=2)
                spec = spec[:1500] + "\n..." if len(spec) > 1500 else spec
                lines.append(f"```json\n{spec}\n```")

            if r.metrics:
                m = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                             for k, v in r.metrics.items() if k in ['mean_diff', 'max_diff', 'tc', 'rmse'])
                if m:
                    lines.append(f"Metrics: {m}\n")

        return "\n".join(lines)

    def get_latest_feedback(self) -> str:
        """Get feedback from the most recent iteration."""
        return self.results[-1].feedback if self.results else ""

    def should_early_stop(self, target_error: float = 0.01, min_iterations: int = 2,
                          no_improvement_patience: int = 3) -> Tuple[bool, str]:
        """Determine if early stopping criteria are met."""
        if len(self.results) < min_iterations:
            return False, ""

        if self.best_error < target_error:
            return True, f"Target achieved: {self.best_error:.6f} < {target_error}"

        if self.best_error < 0.0001:
            return True, f"Near-perfect fit: {self.best_error:.6e}"

        if len(self.results) >= no_improvement_patience:
            recent = [r.error_value for r in self.results[-no_improvement_patience:]]
            if all(e >= self.best_error for e in recent) and max(recent) - min(recent) < 0.001:
                return True, f"No improvement for {no_improvement_patience} iterations"

        return False, ""

    def get_dynamic_error_threshold(self) -> float:
        """Calculate dynamic error threshold based on current best."""
        if self.best_error >= float('inf') or self.best_error <= 0:
            return 0.1
        target = 10 ** np.floor(np.log10(self.best_error))
        return max(target / 10.0 if target >= self.best_error else target, 1e-8)

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