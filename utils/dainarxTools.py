"""
Dainarx Validation Tool for Hybrid Automaton Evaluation

This tool allows the ManagerAgent to directly validate generated HA JSON specifications
against ground truth trajectory data using the Dainarx HAEvaluator.

The tool provides real-time feedback including:
- Simulation metrics (TC, Max Difference, Mean Difference)
- Trajectory comparison plots (base64-encoded)
- Detailed error messages for debugging
"""

import json
import os
import sys
from typing import Any, Dict, Optional
from smolagents.default_tools import Tool

# Add Dainarx_code to path for importing evaluation modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dainarx_code_dir = os.path.join(_current_dir, 'Dainarx_code')
if os.path.exists(_dainarx_code_dir) and _dainarx_code_dir not in sys.path:
    sys.path.insert(0, _dainarx_code_dir)

# Import Dainarx main functions that accept dict directly
from main import main_from_dict, validate_ha_specification as dainarx_validate

# Import HA evaluation module for alternative evaluation path
from HA_evaluation import HAEvaluator

# Import HA specification validator
try:
    from .ha_spec_validator import preprocess_ha_for_evaluation, extract_and_fix_ha_spec
except ImportError:
    from ha_spec_validator import preprocess_ha_for_evaluation, extract_and_fix_ha_spec


class DainarxEvaluationTool(Tool):
    """
    Tool for validating and evaluating Hybrid Automaton specifications against ground truth data.

    This tool enables the ManagerAgent to:
    1. Submit a generated HA JSON specification
    2. Run simulation against ground truth trajectory data
    3. Receive quantitative metrics and qualitative feedback
    4. Iterate and refine the specification based on results
    """

    name = "dainarx_evaluate_ha"
    description = (
        "**Dainarx Hybrid Automaton Evaluation Tool**\n\n"
        "Use this tool to validate your HA specification against ground truth trajectory data.\n"
        "The tool simulates your HA model and compares it to real data.\n\n"
        "**WHAT IT DOES:**\n"
        "1. Validates HA JSON syntax and structure\n"
        "2. Simulates the HA dynamics using Dainarx engine\n"
        "3. Compares simulated trajectory to ground truth data\n"
        "4. Computes evaluation metrics (TC, Max Diff, Mean Diff)\n"
        "5. Returns detailed feedback for refinement\n\n"
        "**METRICS EXPLANATION:**\n"
        "- TC (Change-Point Error): How accurately mode switches are detected (lower is better, <0.01s is good)\n"
        "- Max Difference: Maximum state trajectory error (lower is better, <0.005 is good)\n"
        "- Mean Difference: Average state trajectory error (lower is better)\n\n"
        "**HOW TO USE:**\n"
        "Pass your complete HA specification as a JSON string. The tool returns:\n"
        "- SUCCESS: Metrics showing how well your HA matches the ground truth\n"
        "- FAILURE: Error messages explaining what went wrong\n\n"
        "**TIP:** Use this tool iteratively to refine your HA specification until metrics improve."
    )

    inputs = {
        "ha_spec_json": {
            "type": "string",
            "description": (
                "Your complete Hybrid Automaton specification as a JSON string. "
                "Must include 'automaton' (var, input, mode, edge) and 'config' (dt, total_time) sections. "
                "Example format:\n"
                '{"automaton": {"var": "x1", "input": "u1", "mode": [{"id": 1, "eq": "x1[2] = -0.5 * x1[1] - 5.0 * x1[0] + u1"}], "edge": []}, '
                '"config": {"dt": 0.001, "total_time": 10.0}}'
            )
        }
    }
    output_type = "string"

    def __init__(self,
                 data_path: str = None,
                 worker_agent=None,
                 **kwargs):
        """
        Initialize the Dainarx evaluation tool.

        Args:
            data_path: Path to directory containing .npz ground truth files
            worker_agent: Reference to the parent agent (for accessing shared state)
        """
        super().__init__(**kwargs)
        self.worker_agent = worker_agent

        # Default data path if not provided
        if data_path is None:
            self.data_path = os.path.join(_dainarx_code_dir, 'data_duffing')
        else:
            self.data_path = data_path

        # Verify data path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Find test data file
        npz_files = [f for f in os.listdir(self.data_path) if f.endswith('.npz')]
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_path}")

        # Use first test file (could be made configurable)
        self.npz_file_path = os.path.join(self.data_path, npz_files[0])

        # Store evaluation results for feedback aggregation
        self.evaluation_history = []

    def forward(self, ha_spec_json: str) -> str:
        """
        Evaluate the HA specification against ground truth data.

        This method passes the HA JSON directly to Dainarx's main_from_dict() function,
        enabling real-time validation without requiring pre-written JSON files.

        Args:
            ha_spec_json: HA specification as a JSON string

        Returns:
            Formatted evaluation results with metrics and feedback
        """
        response_parts = []

        # Step 1: Extract and validate the HA specification
        try:
            ha_dict, fix_messages = extract_and_fix_ha_spec(ha_spec_json, auto_fix=True)
        except Exception as e:
            response_parts.append("❌ **EVALUATION FAILED: JSON PARSING ERROR**")
            response_parts.append(f"\nCould not parse your HA specification: {str(e)}")
            response_parts.append("\n**Action Required:** Ensure your input is valid JSON format.")
            return "\n".join(response_parts)

        if ha_dict is None:
            response_parts.append("❌ **EVALUATION FAILED: INVALID SPECIFICATION**")
            response_parts.append("\nCould not extract a valid HA dictionary.")
            response_parts.append("\n**Errors:**")
            for msg in fix_messages:
                response_parts.append(f"  - {msg}")
            return "\n".join(response_parts)

        # Show fixes that were applied
        fixes_applied = [msg for msg in fix_messages if msg.startswith("Fixed:")]
        if fixes_applied:
            response_parts.append("⚠️ **Auto-corrections applied:**")
            for fix in fixes_applied:
                response_parts.append(f"  - {fix.replace('Fixed: ', '')}")
            response_parts.append("")

        # Step 2: Validate structure
        if 'automaton' not in ha_dict or 'config' not in ha_dict:
            response_parts.append("❌ **EVALUATION FAILED: MISSING REQUIRED SECTIONS**")
            response_parts.append("\nYour specification must have both 'automaton' and 'config' sections.")
            return "\n".join(response_parts)

        # Step 3: Run Dainarx main_from_dict() - directly pass dict to main function
        try:
            response_parts.append("🔄 **Running Dainarx Validation...**")
            response_parts.append(f"   (Passing HA JSON directly to main_from_dict())")

            # Call Dainarx main_from_dict - this is the key function that accepts dict directly
            eval_results = main_from_dict(
                ha_dict=ha_dict,
                data_path=self.data_path,
                need_creat=False,  # Don't regenerate data
                need_plot=False
            )

            # Check for errors
            if 'error' in eval_results:
                response_parts.append(f"\n❌ **DAINARX ERROR:** {eval_results['error']}")
                return "\n".join(response_parts)

            # Extract metrics from Dainarx results
            metrics = {
                'tc': eval_results.get('tc'),
                'train_tc': eval_results.get('train_tc'),
                'max_diff': eval_results.get('max_diff'),
                'mean_diff': eval_results.get('mean_diff'),
                'clustering_error': eval_results.get('clustering_error'),
            }

            # Store in history for aggregation
            self.evaluation_history.append({
                'ha_spec': ha_dict,
                'metrics': metrics,
                'success': True,
                'raw_results': eval_results
            })

            # Build success response
            response_parts.append("\n✅ **DAINARX EVALUATION COMPLETE**")
            response_parts.append("\n**Validation Results (from main_from_dict):**")

            # Format metrics
            if metrics.get('tc') is not None:
                tc_val = metrics['tc']
                tc_quality = "excellent" if tc_val <= 0.001 else ("good" if tc_val <= 0.01 else "needs improvement")
                response_parts.append(f"  - TC (Change-Point Error): {tc_val:.6f} seconds ({tc_quality})")
            else:
                response_parts.append("  - TC (Change-Point Error): N/A")

            if metrics.get('max_diff') is not None:
                max_diff = metrics['max_diff']
                max_quality = "excellent" if max_diff < 0.0001 else ("good" if max_diff < 0.005 else "needs improvement")
                response_parts.append(f"  - Max Difference: {max_diff:.6f} ({max_quality})")
            else:
                response_parts.append("  - Max Difference: N/A")

            if metrics.get('mean_diff') is not None:
                mean_diff = metrics['mean_diff']
                response_parts.append(f"  - Mean Difference: {mean_diff:.6f}")
            else:
                response_parts.append("  - Mean Difference: N/A")

            if metrics.get('clustering_error') is not None:
                response_parts.append(f"  - Clustering Error: {metrics['clustering_error']}")

            # Quality assessment
            response_parts.append("\n**Quality Assessment:**")
            if metrics.get('max_diff') is not None and metrics['max_diff'] < 0.005:
                response_parts.append("  Your HA specification produces a good trajectory match!")
                if metrics.get('max_diff') < 0.0001:
                    response_parts.append("  The model is highly accurate - consider this as a final answer.")
            else:
                response_parts.append("  The trajectory has significant deviations from ground truth.")
                response_parts.append("  **Suggestions for improvement:**")
                response_parts.append("  1. Check ODE parameters (coefficients may need adjustment)")
                response_parts.append("  2. Verify mode structure (may need more/fewer modes)")
                response_parts.append("  3. Review guard conditions (switching thresholds)")

            # Show the validated specification
            response_parts.append("\n**Validated Specification:**")
            response_parts.append(f"```json\n{json.dumps(ha_dict, indent=2)}\n```")

            return "\n".join(response_parts)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()

            # Store failed attempt
            self.evaluation_history.append({
                'ha_spec': ha_dict,
                'error': str(e),
                'success': False
            })

            response_parts.append("❌ **DAINARX EVALUATION FAILED**")
            response_parts.append(f"\nError calling main_from_dict(): {str(e)}")
            response_parts.append("\n**Possible Causes:**")
            response_parts.append("  1. Invalid ODE syntax in mode equations")
            response_parts.append("  2. Undefined variables in equations")
            response_parts.append("  3. Invalid guard conditions")
            response_parts.append("  4. Dimension mismatch with ground truth data")
            response_parts.append("\n**Your Specification (for debugging):**")
            response_parts.append(f"```json\n{json.dumps(ha_dict, indent=2)}\n```")

            return "\n".join(response_parts)

    def get_evaluation_summary(self) -> str:
        """
        Get a summary of all evaluations performed during this session.

        Returns:
            Formatted summary of evaluation history
        """
        if not self.evaluation_history:
            return "No evaluations performed yet."

        summary_parts = [f"**Evaluation History ({len(self.evaluation_history)} attempts):**\n"]

        for i, eval_record in enumerate(self.evaluation_history, 1):
            if eval_record['success']:
                metrics = eval_record['metrics']
                max_diff = metrics.get('max_diff', float('inf'))
                summary_parts.append(f"  {i}. SUCCESS - Max Diff: {max_diff:.6f}")
            else:
                summary_parts.append(f"  {i}. FAILED - {eval_record.get('error', 'Unknown error')[:50]}...")

        # Find best result
        successful = [e for e in self.evaluation_history if e['success']]
        if successful:
            best = min(successful, key=lambda x: x['metrics'].get('max_diff', float('inf')))
            best_diff = best['metrics'].get('max_diff', float('inf'))
            summary_parts.append(f"\n**Best Result:** Max Diff = {best_diff:.6f}")

        return "\n".join(summary_parts)


# Convenience function for programmatic evaluation
def evaluate_ha_specification(
    ha_spec: Dict[str, Any],
    data_path: str,
    save_plot: bool = False,
    plot_path: str = None,
    use_dainarx_main: bool = True
) -> Dict[str, Any]:
    """
    Programmatically evaluate an HA specification against ground truth data.

    This function passes the HA dict directly to Dainarx's main_from_dict() function.

    Args:
        ha_spec: HA specification dictionary
        data_path: Path to directory containing .npz files
        save_plot: Whether to save the comparison plot
        plot_path: Path to save the plot (if save_plot=True)
        use_dainarx_main: If True, use main_from_dict(); if False, use HAEvaluator

    Returns:
        Dictionary with 'success', 'metrics', and 'feedback' keys
    """
    if use_dainarx_main:
        # Use Dainarx main_from_dict() - directly passes dict to main function
        try:
            eval_results = main_from_dict(
                ha_dict=ha_spec,
                data_path=data_path,
                need_creat=False,
                need_plot=save_plot
            )

            if 'error' in eval_results:
                return {
                    'success': False,
                    'metrics': {},
                    'feedback': f"Dainarx error: {eval_results['error']}",
                    'ha_spec': ha_spec
                }

            metrics = {
                'tc': eval_results.get('tc'),
                'train_tc': eval_results.get('train_tc'),
                'max_diff': eval_results.get('max_diff'),
                'mean_diff': eval_results.get('mean_diff'),
                'clustering_error': eval_results.get('clustering_error'),
            }

            # Build feedback string
            feedback_parts = [
                f"HA Specification validated via main_from_dict():",
                f"```json\n{json.dumps(ha_spec, indent=2)}\n```",
                f"\nDainarx Results:",
                f"  TC: {metrics['tc']:.6f}" if metrics['tc'] is not None else "  TC: N/A",
                f"  Max Diff: {metrics['max_diff']:.6f}" if metrics['max_diff'] is not None else "  Max Diff: N/A",
                f"  Mean Diff: {metrics['mean_diff']:.6f}" if metrics['mean_diff'] is not None else "  Mean Diff: N/A",
            ]

            return {
                'success': True,
                'metrics': metrics,
                'feedback': "\n".join(feedback_parts),
                'ha_spec': ha_spec,
                'raw_results': eval_results
            }

        except Exception as e:
            import traceback
            return {
                'success': False,
                'metrics': {},
                'feedback': f"main_from_dict() failed: {str(e)}\n{traceback.format_exc()}",
                'ha_spec': ha_spec
            }

    else:
        # Fallback: Use HAEvaluator
        npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        if not npz_files:
            return {
                'success': False,
                'metrics': {},
                'feedback': f"No .npz files found in {data_path}"
            }

        npz_file_path = os.path.join(data_path, npz_files[0])

        try:
            dt = ha_spec['config'].get('dt', 0.001)
            total_time = ha_spec['config'].get('total_time', 10.0)

            evaluator = HAEvaluator(
                ha_dict=ha_spec,
                npz_file_path=npz_file_path,
                dt=dt,
                total_time=total_time
            )

            save_path = plot_path if save_plot else None
            metrics_text, plot_base64 = evaluator(
                plot_mode='overlay',
                save_path=save_path,
                print_metrics=False
            )

            metrics = evaluator.metrics

            feedback_parts = [f"HA Specification:\n```json\n{json.dumps(ha_spec, indent=2)}\n```\n"]
            feedback_parts.append(f"Evaluation Results:\n{metrics_text}")

            return {
                'success': True,
                'metrics': {
                    'tc': metrics.get('tc'),
                    'max_diff': metrics.get('max_diff'),
                    'mean_diff': metrics.get('mean_diff'),
                    'clustering_error': metrics.get('clustering_error')
                },
                'feedback': "\n".join(feedback_parts),
                'ha_spec': ha_spec
            }

        except Exception as e:
            import traceback
            return {
                'success': False,
                'metrics': {},
                'feedback': f"Evaluation failed: {str(e)}\n\nSpecification:\n```json\n{json.dumps(ha_spec, indent=2)}\n```",
                'ha_spec': ha_spec
            }


if __name__ == "__main__":
    # Test the tool
    print("Testing DainarxEvaluationTool...")

    # Create tool instance
    tool = DainarxEvaluationTool()

    # Test with a sample HA specification
    test_spec = json.dumps({
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
            "dim": 2
        }
    })

    print("=" * 60)
    print("Testing evaluation with sample Duffing oscillator spec:")
    print("=" * 60)
    result = tool.forward(test_spec)
    print(result)

    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    print("=" * 60)
    print(tool.get_evaluation_summary())
