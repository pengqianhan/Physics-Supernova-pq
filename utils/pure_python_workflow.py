"""
Pure Python class workflow for HA identification.

Completely removes JSON dependency - direct Python class generation and iteration.
"""

import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
from PIL import Image
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress


def load_and_compress_image(image_path: str, max_short_side_pixels: int = 1080) -> Optional[Image.Image]:
    """Load an image from path and compress it if needed."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        short_side = min(img.size)
        if short_side > max_short_side_pixels:
            scale = max_short_side_pixels / short_side
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        return img
    except Exception:
        return None


@dataclass
class IterationFeedback:
    """Structured feedback from one iteration."""
    iteration: int
    analysis_process: str
    class_code: str
    metrics: dict
    llm_critique: str
    error_value: float
    plot_path: str = ""  # Path to visualization plot for this iteration


def format_iteration_feedback(fb: IterationFeedback) -> str:
    """Format single iteration feedback for inclusion in task."""
    is_failed = fb.error_value < 0 or fb.llm_critique.startswith("[FAILED]")
    status = "FAILED" if is_failed else f"error={fb.error_value:.4f}"

    # Key metrics only
    metrics_str = ", ".join([
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in fb.metrics.items()
        if k in ['max_diff', 'mean_diff', 'tc']
    ]) or "N/A"

    # Truncate code more aggressively
    code_truncated = fb.class_code[:600] + "\n# ..." if len(fb.class_code) > 600 else fb.class_code
    if not code_truncated:
        code_truncated = "# (No code)"

    # Visualization reference
    viz_str = f" | Plot: <iter_{fb.iteration}_plot>" if fb.plot_path else ""

    return f"""### Iter {fb.iteration} [{status}] {metrics_str}{viz_str}
```python
{code_truncated}
```
**Critique**: {fb.llm_critique}
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

    # Add data and tools info (concise)
    task += f"""## Data
- Plots: {image_placeholder_list}
- NPZ files: {npz_placeholder_list}

"""
    if tools_list:
        task += f"**Tools**: {', '.join(tools_list)}\n\n"
        task += "**Tool Usage Hints**:\n"
        if "hybrid_automaton_image_analysis" in tools_list:
            task += "- `hybrid_automaton_image_analysis`: Call this tool to extract detailed information from trajectory images, such as numerical values, system behavior patterns, and mode switching characteristics.\n"
        if "validate_hybrid_automaton_specification" in tools_list:
            task += "- `validate_hybrid_automaton_specification`: Call this tool to validate the structure and syntax of your HA specification.\n"
        task += "\n"

    # Add initial template for first iteration
    if iteration == 1:
        initial_template = get_simple_ha_template(num_variables, num_inputs)
        task += f"""## Template
Fill in: `num_modes()`, `mode_dynamics()`, `guard_condition()`, `reset_map()`

```python
{initial_template}
```

"""

    # Add feedback from previous iterations
    eval_plot_images = []
    if feedback and len(feedback) > 0:
        task += "## Previous Attempts\n\n"
        for fb in feedback:
            task += format_iteration_feedback(fb)
            task += "\n"
            if fb.plot_path:
                eval_img = load_and_compress_image(fb.plot_path)
                if eval_img:
                    eval_plot_images.append(eval_img)

    # Combine trace images with evaluation plot images
    # Order: trace images first, then evaluation plots (matching placeholder order in task)
    all_images = compressed_trace_images + eval_plot_images

    # Final instructions
    var_str = ', '.join([f'x{i+1}' for i in range(num_variables)])
    input_str = ', '.join([f'u{i+1}' for i in range(num_inputs)]) if num_inputs > 0 else ''
    task += f"""
## Output
Generate complete Python class. Keep `self.var = "{var_str}"` and `self.input = "{input_str}"` FIXED.

{STRUCTURED_OUTPUT_FORMAT}
"""

    return task, all_images


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
