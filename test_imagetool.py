"""Simple test for ReviewRequestTool_ha"""
import sys
from pathlib import Path

# Add project root to sys.path so we can import utils
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.imgTools_ha import HybridAutomatonImageTool
from utils.markdown_utils import MarkdownMessage, load_trace_data_from_filepath
import json
# vision_model_id = 'openai/kimi-k2.5'
vision_model_id = 'gemini/gemini-3-flash-preview'
# Create a mock worker_agent with required attributes
class MockWorkerAgent:
    def __init__(self):
        # Create a proper MarkdownMessage object (not just a string)
        # The content should be a list of dicts with "type" and "text"/"image_url" keys
        self.markdown_content_high_res_image = load_trace_data_from_filepath(str(project_root / "data_all/non_linear/duffing"))
# Set high res images in the agent, for the AskImageTool to use
# Dynamically add custom attribute to agent for data sharing with tools
# Docs: https://huggingface.co/docs/smolagents/en/tutorials/building_good_agents (agents support dynamic attributes)
# Create the review tool with mock worker agent
mock_agent = MockWorkerAgent()
image_tool = HybridAutomatonImageTool(worker_agent=mock_agent,vision_model_id=vision_model_id)


success, error_msg = image_tool.register_iteration_image(100, "evaluation_results_gemini3flash_260129/ATVA/ball/runs/20260125_174104_11ed/iter_1/overlay.png")
if not success:
    print(f"Registration failed: {error_msg}")
    sys.exit(1)

feedback = image_tool.forward(
    image_ref="<iter_image_1_0>",
    question="What is the title of the image?"
)

print("\n" + "=" * 60)
print("Image Analysis Feedback:")
print("=" * 60)
print(feedback)
