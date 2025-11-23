import os
import time
import json
from typing import Any, Dict, List, Optional
from smolagents.default_tools import Tool

# Import smolagents components for LLM model and message handling
from smolagents import (
    LiteLLMModel
)
from smolagents.models import ChatMessage, MessageRole
from .markdown_utils import MarkdownMessage, compressed_image_content

# Import evaluation module
from .Dainarx_code.HA_evaluation import HAEvaluator


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

    def __init__(self, worker_agent=None, review_tool_model: str = "gemini/gemini-2.5-flash-lite"):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to main agent for accessing problem context
        gemini_apikey = os.getenv('GEMINI_API_KEY')
        self.review_model = LiteLLMModel(model_id=review_tool_model, api_key=gemini_apikey)
        self.npz_file_path = "data_duffing/test_data0.npz"
    def forward(self, my_ha_solution: Dict[str, Any], my_note: str) -> str:
        """Request expert review of hybrid automaton specification with focus areas and return detailed feedback."""
        if not self.worker_agent or not hasattr(self.worker_agent, "markdown_content_high_res_image"):
            return "Error: No markdown content available."
        
        evaluator = HAEvaluator(
        ha_dict=my_ha_solution,
        npz_file_path=self.npz_file_path,
        dt=0.001,
        total_time=10.0)
        metrics_text, plot_base64 = evaluator(
        plot_mode="single",
        save_path='data_duffing_evaluation/output_single.png',
        print_metrics=True)
        