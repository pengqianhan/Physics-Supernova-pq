"""Simple test for ReviewRequestTool_ha"""
from utils.reviewTools_ha import ReviewRequestTool_ha
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
review_tool = ReviewRequestTool_ha(worker_agent=mock_agent)

# Test HA dict
ha_dict = {
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
        "dim": 2,
        "other_items": ""
    }
}

print("=" * 60)
print("Testing ReviewRequestTool_ha")
print("=" * 60)
print(f"\nHA Solution:\n{json.dumps(ha_dict, indent=2)}")
print("\n" + "-" * 60)
print("Calling review tool...")
print("-" * 60 + "\n")

feedback = review_tool.forward(
    my_ha_solution=json.dumps(ha_dict, indent=2),
    my_note="Please check if the mode equations are correct and verify transition conditions"
)

print("\n" + "=" * 60)
print("Review Feedback:")
print("=" * 60)
print(feedback)
