"""
Pure Python class workflow for HA identification.

Completely removes JSON dependency - direct Python class generation and iteration.
"""

import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress


@dataclass
class IterationFeedback:
    """Structured feedback from one iteration."""
    iteration: int
    analysis_process: str
    class_code: str
    metrics: dict
    llm_critique: str
    error_value: float


def format_iteration_feedback(fb: IterationFeedback) -> str:
    """Format single iteration feedback for inclusion in task."""
    # Check if this is a failed iteration
    is_failed = fb.error_value < 0 or fb.llm_critique.startswith("[FAILED]")

    # Format header based on success/failure
    if is_failed:
        header = f"### Iteration {fb.iteration} [FAILED]"
    else:
        header = f"### Iteration {fb.iteration} (Error: {fb.error_value:.6f})"

    # Format metrics
    metrics_str = ", ".join([
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in fb.metrics.items()
        if k in ['max_diff', 'mean_diff', 'tc', 'rmse']
    ]) or "(No metrics - evaluation failed)"

    # Truncate analysis and code
    analysis_truncated = fb.analysis_process[:300] + "..." if len(fb.analysis_process) > 300 else fb.analysis_process
    if not analysis_truncated:
        analysis_truncated = "(No analysis provided)"

    code_truncated = fb.class_code[:1000] + "\n# ... (truncated)" if len(fb.class_code) > 1000 else fb.class_code
    if not code_truncated:
        code_truncated = "# (No code extracted)"

    return f"""{header}

**Analysis**: {analysis_truncated}

**Code**:
```python
{code_truncated}
```

**Metrics**: {metrics_str}

**Expert Critique**: {fb.llm_critique}
"""


def generate_pure_python_task(
    input_data_path: str,
    num_variables: int,
    num_inputs: int,
    iteration: int = 1,
    feedback: Optional[List[IterationFeedback]] = None,
    tools_list: List[str] = None,
    manager_type: str = "CodeAgent"
) -> Tuple[str, list]:
    """
    Generate task prompt for pure Python class-based HA generation.

    NO JSON Schema, NO conversion - just Python class.

    Args:
        input_data_path: Path to trace data directory
        num_variables: Number of state variables
        num_inputs: Number of input variables
        iteration: Current iteration number
        feedback: List of IterationFeedback from previous iterations
        tools_list: Available tools
        manager_type: Type of agent

    Returns:
        task: Task prompt string
        images: Compressed trajectory images
    """
    from prompts_ha.prompts_pure_python import (
        get_python_class_task_prefix,
        STRUCTURED_OUTPUT_FORMAT
    )
    from utils.ha_class_template import get_simple_ha_template

    # Load trace data
    markdown_content = load_trace_data_from_filepath(input_data_path)
    image_placeholder_list = list(markdown_content.image_paths.keys())
    npz_placeholder_list = list(markdown_content.npz_paths.keys())

    # Get compressed images for LLM
    compressed_trace_images = markdown_images_compress(markdown_content, max_short_side_pixels=1080)

    # Build task prompt
    task = get_python_class_task_prefix(num_variables, num_inputs, iteration)

    # Add tool/agent information
    task += "\n## Available Resources\n"

    if tools_list and len(tools_list) > 0:
        task += f"**Tools**: You have access to: {', '.join(tools_list)}\n"
        task += "- Use `hybrid_automaton_image_analysis` to get vision model insights on trajectory patterns\n"
        task += "- Use `validate_hybrid_automaton_specification` to check your class syntax before submission\n\n"

    if manager_type == "CodeAgent":
        task += "**Code Execution**: You can execute Python code to analyze data or test ideas\n\n"

    # Add data sources
    task += f"""## Data Sources
- **Trajectory plots**: {image_placeholder_list}
- **Raw data files**: {npz_placeholder_list}

"""

    # Add initial template for first iteration
    if iteration == 1:
        initial_template = get_simple_ha_template(num_variables, num_inputs)
        task += f"""## Starting Template (v0)

Here's a minimal template with the correct structure. Your job is to fill in the correct:
- Number of modes (`num_modes()`)
- Dynamics equations (`mode_dynamics()`)
- Guard conditions (`guard_condition()`)
- Reset logic (`reset_map()` if needed)

```python
{initial_template}
```

**Replace the TODO placeholders with your inferred dynamics!**

"""

    # Add feedback from previous iterations (with LLM critique)
    if feedback and len(feedback) > 0:
        task += "\n## Previous Iterations (with Expert Critique)\n\n"
        task += "Learn from these previous attempts and their expert critiques:\n\n"
        for fb in feedback:
            task += format_iteration_feedback(fb)
            task += "\n---\n\n"
        task += "Use the critiques above to improve your next attempt!\n\n"

    # Final instructions with structured output format
    var_str = ', '.join([f'x{i+1}' for i in range(num_variables)])
    input_str = ', '.join([f'u{i+1}' for i in range(num_inputs)]) if num_inputs > 0 else ''
    task += f"""
## Your Output

Generate a **complete, executable Python class** that:
1. Accurately models the observed dynamics
2. Uses `self.params` for all tunable numerical values
3. Implements all required methods (`__init__`, `num_modes`, `mode_dynamics`, `guard_condition`, `reset_map`)

**Important**:
- Keep `self.var = "{var_str}"` FIXED
- Keep `self.input = "{input_str}"` FIXED
- Provide reasonable initial guesses for `self.params` (they will be auto-optimized)

{STRUCTURED_OUTPUT_FORMAT}
"""

    return task, compressed_trace_images


# Quick test
if __name__ == "__main__":
    task, images = generate_pure_python_task(
        input_data_path="data_all/non_linear/duffing",
        num_variables=1,
        num_inputs=1,
        iteration=1,
        feedback="",
        tools_list=["hybrid_automaton_image_analysis"],
        manager_type="CodeAgent"
    )

    print("=" * 80)
    print("GENERATED TASK PROMPT")
    print("=" * 80)
    print(task)
    print("\n" + "=" * 80)
    print(f"Number of images: {len(images)}")
