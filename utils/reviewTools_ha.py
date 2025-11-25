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
HA_REVIEW_SYSTEM_PROMPT = """You are an expert hybrid automata reviewer specializing in dynamical systems modeling. Your job is to critically evaluate hybrid automaton specifications and provide actionable feedback.

## Your Review Responsibilities:

1. **Mode Dynamics Validation**:
   - Check if the differential equations for each mode correctly model the physical behavior
   - Verify dimensional consistency in all equations
   - Look for sign errors, missing terms, or incorrect coefficients

2. **Transition Conditions**:
   - Validate guard conditions for mode switches
   - Check if transition conditions are physically meaningful
   - Verify that guards properly capture the switching behavior shown in the data

3. **State Variables and Inputs**:
   - Ensure all state variables are properly defined
   - Verify input signals are correctly incorporated into dynamics
   - Check for missing state variables or inputs

4. **Metrics Interpretation**:
   - The evaluation metrics show how well the HA matches ground truth data
   - `tc` (change-point error): How accurately mode switches are detected (lower is better)
   - `max_diff` / `mean_diff`: Maximum and mean state trajectory differences (lower is better)
   - Use these metrics to identify specific areas needing improvement

5. **Visualization Analysis**:
   - Compare the simulation plot against the ground truth
   - Identify time regions where the model diverges from data
   - Note any mode mismatches visible in the trajectories

## Output Format:
Provide structured feedback with:
- **Critical Errors**: Issues that must be fixed
- **Warnings**: Potential problems to investigate
- **Suggestions**: Improvements that could help
- **Metrics Analysis**: Interpretation of the quantitative evaluation

Be specific and actionable. Reference the metrics and visualization in your feedback.
"""


class ReviewRequestTool_ha(Tool):
    """Hybrid Automata expert reviewer to provide critical reviews and feedback on HA specifications."""

    name = "ask_ha_review_expert"
    description = (
        "Request expert review of your hybrid automaton specification. Provide your HA Python dict and what you want reviewed. "
        "You should parse-in your current HA Python dict through 'my_ha_solution' input (or the reviewer would not be able to see it). "
        "The reviewer will provide detailed feedback to help improve your hybrid automaton specification."
    )
    inputs = {
        "my_ha_solution": {
            "type": "string",
            "description": "Your current hybrid automaton specification (Python dict) that needs review. This must be provided clearly and completely."
        },
        "my_note": {
            "type": "string",
            "description": "What aspects to focus on (e.g., 'Check mode equations', 'Verify transition conditions', 'Overall review'), or your note/uncertain points/things you feel may go wrong."
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
        self.npz_file_path = "data_duffing/test_data0.npz"

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
            f"## Hybrid Automaton Solution (Python Dict) Under Review:\n"
            f"```json\n{json.dumps(ha_dict, indent=2)}\n```\n\n"
            f"## Evaluation Metrics:\n{metrics_text}\n\n"
            f"## Agent's Note:\n{my_note}\n\n"
            f"## Original Problem Context:\n"
            f"The following images show the original problem and data.\n"
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
                "text": "\n## Simulation vs Ground Truth Plot:\nThe following plot shows the HA simulation (predicted) overlaid with ground truth data."
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