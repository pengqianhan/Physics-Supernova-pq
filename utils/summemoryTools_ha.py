import os
import time
from smolagents.default_tools import Tool
from copy import deepcopy

# Import smolagents components for LLM model, memory, and agent handling
from smolagents import (
    LiteLLMModel
)
from smolagents.models import ChatMessage, MessageRole
from smolagents.memory import AgentMemory
from smolagents.agents import MultiStepAgent

# Debug flag for development and troubleshooting
DEBUGGING = False



# Extract agent's memory observation for summarization (similar to smolagents/memory.py/AgentMemory/replay)
def get_agent_memory_observation(agent: MultiStepAgent) -> str:
    """Extract the agent's complete Hybrid Automaton learning and refinement history from memory."""

    # Get the complete message history from the agent's memory
    newest_input_messages = agent.write_memory_to_messages().copy()
    agent_observation_and_action_str = ""
    agent_observation_and_action_str += f"You were given this task: {newest_input_messages[1]}\nHere is your previous Hybrid Automaton learning process: <Your previous HA refinement process>"
    # Note: model_input_message[1] should be the task
    agent_observation_and_action_str += str(newest_input_messages[2:]) + "</Your previous HA refinement process>"
    # Note: model_input_message[0] is system prompt, [1] is first task; [2:] is model response & tool response
    return agent_observation_and_action_str
    
def summarize_agent_memory(summarize_model: LiteLLMModel,
                           agent: MultiStepAgent = None) -> str:
    """Generate a comprehensive summary of the agent's HA learning and refinement iterations."""
    agent_observation = get_agent_memory_observation(agent=agent)

    # System prompt for the summarization model to ensure comprehensive summary of HA iterations
    system_prompt = (
        "You are here to SUMMARIZE your iterative work on learning and refining a Hybrid Automaton specification from trace data. "
        "You will be given your previous actions on the Hybrid Automaton learning task. Your goal is to provide a complete history of ALL Hybrid Automaton versions you generated, "
        "starting from the initial specification (v0) through all refinement iterations (v1, v2, v3, ...) up to the latest version. "
        "\n\n"
        "For EACH version, you must include:\n"
        "1. **Version number** (v0, v1, v2, etc.)\n"
        "2. **Complete Hybrid Automaton specification** (the full Python dict with 'automaton' and 'config' sections)\n"
        "3. **Analysis/Reasoning** that led to this version:\n"
        "   - For v0: Initial understanding or assumptions\n"
        "   - For v1+: What problems were found in the previous version, what metrics/feedback indicated issues, and what changes were made to address them\n"
        "4. **Evaluation metrics** (if available): tc, max_diff, mean_diff, etc.\n"
        "5. **Review feedback** (if you used the review tool): key insights from the expert reviewer\n"
        "\n"
        "IMPORTANT:\n"
        "- Do NOT omit any Hybrid Automaton version you created during the refinement process\n"
        "- Include the COMPLETE Python dict for each version (not just the changes, include the whole dict)\n"
        "- Your summary should trace the evolution from v0 to the latest version, showing how each iteration improved upon the previous one\n"
        "- Make sure the dict format is valid Python dict (proper quotes, commas, brackets)\n"
    )

    # Combine agent observation with output format template (task requirements already in system_prompt)
    combined_prompt = agent_observation + "\n\n" + (
        "Now, summarize ALL Hybrid Automaton specification versions using the following format:\n\n"
        "## Version v0 (Initial Specification)\n"
        "### Hybrid Automaton Specification:\n"
        "```python\n"
        "{...complete Python dict...}\n"
        "```\n"
        "### Analysis:\n"
        "[Initial understanding, assumptions, or baseline approach]\n"
        "### Metrics:\n"
        "[Metrics if evaluated, or 'Not evaluated yet']\n"
        "### Review Feedback:\n"
        "[Feedback if reviewed, or 'Not reviewed yet']\n"
        "\n"
        "## Version v1\n"
        "### Hybrid Automaton Specification:\n"
        "```python\n"
        "{...complete Python dict...}\n"
        "```\n"
        "### Analysis:\n"
        "[What problems were found in v0, what metrics/feedback indicated, what changes were made]\n"
        "### Metrics:\n"
        "[Metrics from evaluation]\n"
        "### Review Feedback:\n"
        "[Feedback from expert reviewer]\n"
        "\n"
        "## Version v2\n"
        "...\n"
        "\n"
        "Please provide the complete iteration history from v0 to the latest version:\n"
    )
    
    # Debug mode: save agent observation to file for inspection
    if DEBUGGING:
        with open("sum_input_agent_observation_now.txt","w") as f:
            f.write(agent_observation)
    
    # Prepare messages for summarization model
    messages = [
        ChatMessage(role=MessageRole.SYSTEM,content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=combined_prompt),
    ]
    
    # Generate summary using the summarization model
    response = summarize_model.generate(messages)
    summary = response.content
    return summary

class SummarizeMemoryTool(Tool):
    """Tool to summarize the complete Hybrid Automaton learning iteration history from v0 to the latest version."""

    name = "summarize_hybrid_automaton_iterations"

    description = (
        "When you have completed multiple iterations of Hybrid Automaton specification refinement and want to get a comprehensive summary of ALL versions (v0, v1, v2, ...), call this tool. "
        "This tool will generate a complete history showing:\n"
        "- Each Hybrid Automaton version (v0 = initial, v1, v2, ... = refined versions)\n"
        "- The complete JSON specification for each version\n"
        "- Analysis explaining what led to each version (problems found, metrics, changes made)\n"
        "- Evaluation metrics for each version\n"
        "- Review feedback received for each version\n"
        "\n"
        "This tool automatically extracts all Hybrid Automaton versions from your memory, so you don't need to pass in the specifications explicitly. "
        "Use this when you want to:\n"
        "1. Review the complete evolution of your Hybrid Automaton learning process\n"
        "2. Document all iterations before finalizing\n"
        "3. Get a structured summary showing progress from v0 to the current version\n"
        "\n"
        "Optional: Set reset_memory=True to clear the memory after summarization (useful for genetic algorithm-based evolution where you want to start fresh iterations).\n"
    )

    inputs = {
        "reset_memory": {
            "type": "boolean",
            "description": "Whether to reset agent memory after summarization. Set to True for genetic algorithm scenarios where you want to start a fresh iteration. Default is False.",
            "default": False,
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self, worker_agent=None, summarize_model_id: str = "gemini/gemini-2.5-flash-lite"):
        super().__init__()
        self.worker_agent = worker_agent  # Reference to the main agent for memory access
        api_key = os.environ.get("GEMINI_API_KEY")
        # Initialize summarization model with higher token limit for comprehensive HA iteration summaries
        self.summarize_model = LiteLLMModel(
            model_id=summarize_model_id,
            api_key=api_key,
            max_completion_tokens=16384,
            num_retries=3,
            timeout=1200,
        )
        self.is_first_use = True  # Track first use to modify task instructions
    def forward(self, reset_memory: bool = False) -> str:  # type: ignore[override]
        """Execute Hybrid Automaton iteration history summarization and optionally reset agent memory.

        Args:
            reset_memory: If True, reset agent memory after summarization.
                         Useful for genetic algorithm scenarios where each generation
                         starts fresh. Default is False.
        """
        if not self.worker_agent or not hasattr(self.worker_agent, "memory"):
            return "Error: No memory available."

        memory: AgentMemory = self.worker_agent.memory

        # Retry logic for robust summarization
        max_try = 3
        for _ in range(max_try):
            summary = summarize_agent_memory(self.summarize_model, agent=self.worker_agent)

            if summary is None or summary.strip() == "":
                time.sleep(5)  # Wait before retry
            else:
                break
        if summary is None or summary.strip() == "":
            return "Error: No summary generated after multiple attempts. Please try again later."

        # Optional: Reset memory for genetic algorithm scenarios
        if reset_memory:
            # Store the initial task before resetting
            new_memory_step = deepcopy(memory.steps[0])
            # Note: memory.steps[0] is the initial task

            if self.is_first_use:
                # Update task instructions for subsequent iterations
                new_memory_step.task += (
                    "\n\nYou have completed a generation of Hybrid Automaton learning iterations and summarized your work. "
                    "The summary has been recorded. Now you will start a fresh iteration (new generation in genetic algorithm). "
                    "You may build upon insights from previous generations, but your memory has been reset for a clean start. "
                    "When calling the review tool, remember to parse in your current Hybrid Automaton specification through 'my_ha_solution' input."
                )

            # Clear memory and set up fresh start with context
            self.worker_agent.memory.reset()
            self.worker_agent.memory.steps = []
            self.worker_agent.memory.steps.append(new_memory_step)
            self.is_first_use = False

            memory_status = "\n\n**[Memory Reset]** Agent memory has been reset for a fresh iteration. Ready to start new generation."
        else:
            memory_status = "\n\n**[Memory Preserved]** Agent memory retained. You can continue refining based on the complete historical context."

        # Format the summary of all HA iterations for agent reference
        this_tool_return_prompt = f"""## Complete Hybrid Automaton Learning Iteration History

The following is a comprehensive summary of ALL Hybrid Automaton specification versions you generated during the learning process, from the initial specification (v0) through all refinement iterations up to the latest version.

---

{summary}

---

This completes the summary of your Hybrid Automaton learning iterations from v0 to the latest version. You may use this summary to:
1. Review the complete evolution of your Hybrid Automaton specifications
2. Identify patterns in what improvements worked best
3. Document your learning process
4. Continue refining based on this historical context{memory_status}"""

        return this_tool_return_prompt