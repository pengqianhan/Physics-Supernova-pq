"""
Pure Python class workflow for HA identification.

Completely removes JSON dependency - direct Python class generation and iteration.
"""

import os
from typing import Tuple, List
from utils.markdown_utils import load_trace_data_from_filepath, markdown_to_plaintext, markdown_images_compress


def generate_pure_python_task(
    input_data_path: str,
    num_variables: int,
    num_inputs: int,
    iteration: int = 1,
    feedback: str = "",
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
        feedback: Feedback from previous iteration
        tools_list: Available tools
        manager_type: Type of agent

    Returns:
        task: Task prompt string
        images: Compressed trajectory images
    """
    from prompts_ha.prompts_pure_python import get_python_class_task_prefix, get_feedback_section
    from utils.ha_class_template import get_simple_ha_template

    # Load trace data
    markdown_content = load_trace_data_from_filepath(input_data_path)
    image_paths_list = list(markdown_content.image_paths.keys())
    npz_paths_list = list(markdown_content.npz_paths.keys())

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
- **Trajectory plots**: {len(image_paths_list)} visualization(s) available
- **Raw data files**: {npz_paths_list}

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

    # Add feedback from previous iteration
    if feedback:
        task += get_feedback_section(feedback)

    # Final instructions
    task += f"""
## Your Output

Generate a **complete, executable Python class** that:
1. Accurately models the observed dynamics
2. Uses `self.params` for all tunable numerical values
3. Implements all required methods (`__init__`, `num_modes`, `mode_dynamics`, `guard_condition`, `reset_map`)

**Important**:
- Keep `self.var = "{', '.join([f'x{i+1}' for i in range(num_variables)])}"` FIXED
- Keep `self.input = "{', '.join([f'u{i+1}' for i in range(num_inputs)]) if num_inputs > 0 else ''}"` FIXED
- Provide reasonable initial guesses for `self.params` (they will be auto-optimized)

Output only the Python class code, no explanations before or after.
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
