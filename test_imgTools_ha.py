"""Simple test for ReviewRequestTool_ha"""
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.markdown_utils import MarkdownMessage, load_trace_data_from_filepath
import json


# Create a mock worker_agent with required attributes
class MockWorkerAgent:
    def __init__(self):
        # Create a proper MarkdownMessage object (not just a string)
        # The content should be a list of dicts with "type" and "text"/"image_url" keys
        self.markdown_content_high_res_image = load_trace_data_from_filepath("utils/Dainarx_code/data_duffing")
# Set high res images in the agent, for the AskImageTool to use
# Dynamically add custom attribute to agent for data sharing with tools
# Docs: https://huggingface.co/docs/smolagents/en/tutorials/building_good_agents (agents support dynamic attributes)
# Create the review tool with mock worker agent
mock_agent = MockWorkerAgent()
image_tool = HybridAutomatonImageTool(worker_agent=mock_agent)


feedback = image_tool.forward(
    image_ref="<image_0>",
    question="What is the number of samples in the image?"
)

print("\n" + "=" * 60)
print("Image Analysis Feedback:")
print("=" * 60)
print(feedback)
