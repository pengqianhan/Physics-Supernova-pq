import os
import time
import json
import openai
from typing import Any, Dict, List
from smolagents.default_tools import Tool
from .markdown_utils import MarkdownMessage, compressed_image_content
from dotenv import load_dotenv
load_dotenv()
# Import evaluation module
from .Dainarx_code.HA_evaluation import HAEvaluator


# System prompt for critical hybrid automata review
HA_REVIEW_SYSTEM_PROMPT = """# Expert Hybrid Automaton Specification Reviewer

You are a **Senior Control Systems Engineer** with deep expertise in hybrid dynamical systems, system identification, and formal verification. Your task is to critically evaluate hybrid automaton (HA) specifications and provide actionable, technically rigorous feedback.

## Review Framework

### 1. MODE DYNAMICS VALIDATION ("automaton['mode']")
**Focus Areas ("automaton['mode'][i]['eq'] for i in range(len(automaton['mode']))"):**
- **ODE Correctness**: Verify differential equations match observed trajectory dynamics

### 2. TRANSITION LOGIC VALIDATION ("automaton['edge']")
**Guard Conditions ("automaton['edge'][i]['condition'] for i in range(len(automaton['edge']))"):**

**Reset Maps ("automaton['edge'][i]['reset'] for i in range(len(automaton['edge']))"):**

**Transition Graph ("edge['direction']"):**

### 3. QUANTITATIVE METRICS INTERPRETATION
**Evaluation Metrics ("tc", "max_diff", "mean_diff"):**
- `tc` (Mode switch detection accuracy): < 0.01s is good, < 0.001s is excellent
- `max_diff` (worst-case trajectory deviation): < 0.02 is good, < 0.001 is excellent
- `mean_diff` (average trajectory error): < 0.005 is good, < 0.0001 is excellent


### 4. VISUALIZATION CROSS-CHECK
**Simulation vs Ground Truth Plot Analysis:**

## Output Format:
Provide structured feedback with:
- **Critical Errors**: Issues that must be fixed
- **Warnings**: Potential problems to investigate
- **Suggestions**: Improvements that could help
- **Metrics Interpretation**: Detailed interpretation of quantitative evaluation results.

---
**Remember**: Be specific, cite evidence from metrics/plots, and provide actionable recommendations with concrete values when possible.
"""


class ReviewRequestTool_ha(Tool):
    """Expert HA specification reviewer with automated evaluation and structured feedback generation."""

    name = "ask_ha_review_expert"
    description = (
        "**Expert Hybrid Automaton Python dict Specification Review Service**\n\n"
        "This tool provides comprehensive expert review of your HA specification by:\n"
        "1. **Automated Evaluation**: Simulates your HA against ground truth data and computes accuracy metrics\n"
        "2. **Visual Comparison**: Generates overlay plot comparing your simulation to actual trajectory\n"
        "3. **Expert Analysis**: Senior control systems engineer reviews your spec with structured feedback\n\n"
        "**HOW TO USE:**\n"
        "- `my_ha_solution`: Your complete Hybrid Automaton specification as a Python dict (REQUIRED - the reviewer cannot see your spec otherwise!)\n"
        "- `my_note`: Focus areas or specific concerns (e.g., 'Unsure about mode 2 dynamics', 'Guard thresholds may be wrong')\n\n"
        "**OUTPUT INCLUDES:**\n"
        "- Critical errors that must be fixed\n"
        "- Warnings to investigate\n"
        "- Suggestions for improvement\n"
        "- Detailed metrics interpretation (tc, max_diff, mean_diff)\n"
    )
    inputs = {
        "my_ha_solution": {
            "type": "string",
            "description": (
                "Your complete Hybrid Automaton specification as a Python dict. Must include 'automaton' (with var, input, mode, edge) "
                "and 'config' (with dt, total_time, dim, need_reset, non_linear_items). Example format:\n"
                '{"automaton": {"var": "x1", "input": "u1", "mode": [...], "edge": [...]}, "config": {...}}'
            )
        },
        "my_note": {
            "type": "string",
            "description": (
                "Specific aspects to focus on or your concerns/uncertainties."
            )
        },
    }
    output_type = "string"

    def __init__(self, worker_agent=None, review_model_id: str = "gemini-2.5-flash-lite"):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to main agent for accessing problem context
        # Initialize OpenAI client for Gemini API (same as imgTools_ha.py)
        self.client = openai.OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY") if os.getenv("GEMINI_API_KEY") else "<<<<<<your_api_key>>>>>>>",
        )
        self.review_model_id = review_model_id
        self.npz_file_path = "data_duffing/sample_0.npz"

    def forward(self, my_ha_solution: str, my_note: str) -> str:  # type: ignore[override]
        """Request expert review of hybrid automaton specification with focus areas and return detailed feedback."""

        # Step 1: Validate worker_agent context
        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            # return "Error: No markdown content available."
            print("Error: No markdown content available.")

        # Step 2: Parse HA solution JSON string to dict
        try:
            ha_dict: Dict[str, Any] = json.loads(my_ha_solution)
        except json.JSONDecodeError as e:
            return f"Error: Failed to parse HA solution as JSON: {e}"

        # Step 3: Run HAEvaluator to get metrics and simulation plot
        try:
            evaluator = HAEvaluator(
                ha_dict=ha_dict,
                npz_file_path=self.npz_file_path,
                dt=0.001,
                total_time=10.0
            )
            metrics_text, plot_base64 = evaluator(
                plot_mode="overlay",
                save_path=None,
                print_metrics=False
            )
        except Exception as e:
            return f"Error: HAEvaluator failed: {e}"

        # Step 4: Get compressed version of problem images for review context
        markdown_content: MarkdownMessage = compressed_image_content(
            self.worker_agent.markdown_content_high_res_image
        )

        # Step 5: Build unified review instruction combining HA solution + metrics + user note
        review_instruction = (
            f"# HYBRID AUTOMATON REVIEW REQUEST\n\n"
            f"## Hybrid Automaton Specification Under Review\n"
            f"```python\n{json.dumps(ha_dict, indent=2)}\n```\n\n"
            f"## Evaluation Metrics:\n{metrics_text}\n\n"
            f"## Agent's Note:\n{my_note}\n\n"
        )

        # Step 6: Combine content - review text + original problem images + simulation plot
        # combined_content: List[Dict[str, Any]] = [
        #     {"type": "text", "text": review_instruction}
        # ] + markdown_content.content
        combined_content: List[Dict[str, Any]] = [
            {"type": "text", "text": review_instruction}
        ]

        # Add simulation plot visualization for comparison
        if plot_base64:
            combined_content.append({
                "type": "text",
                "text": "\n## Simulation vs Ground Truth Plot:\nThe following plot shows the Hybrid Automaton simulation (predicted) overlaid with ground truth data."
            })
            combined_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{plot_base64}"}
            })

        # Step 7: Prepare messages for review model (OpenAI format)
        messages = [
            {
                "role": "system",
                "content": HA_REVIEW_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": combined_content
            }
        ]

        # Step 8: Retry logic for robust review generation
        max_try = 3
        last_error = None

        for attempt in range(max_try):
            try:
                response = self.client.chat.completions.create(
                    model=self.review_model_id,
                    messages=messages,
                    max_tokens=8192,
                )
                if response.choices[0].message.content and response.choices[0].message.content.strip():
                    return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                print(f"Review model attempt {attempt + 1}/{max_try} failed: {e}")

            # Wait before retry (but not after the last attempt)
            if attempt < max_try - 1:
                time.sleep(5)

        # Step 9: Fallback response for failed review attempts
        if last_error is not None:
            return f"Error: Review model failed after {max_try} attempts: {last_error}"

        # If no exception but returned is None or empty
        return "No response from review model."