# Import all custom tools available for the hybrid automaton agent
from .markdown_utils import MarkdownMessage, load_markdown_from_filepath, markdown_to_plaintext, markdown_images_compress  # Markdown processing utilities
from .imgTools_ha import HybridAutomatonImageTool  # Image analysis tool for hybrid automaton
from .reviewTools_ha import ReviewRequestTool_ha  # Expert review tool for hybrid automaton
from .summemoryTools_ha import SummarizeMemoryTool  # Memory summarization tool for hybrid automaton
from .validateTools_ha import ValidateHASpecTool  # Validation tool for hybrid automaton
from .dainarxTools import DainarxEvaluationTool, evaluate_ha_specification  # Dainarx evaluation tool
# Dependency check: make sure tenacity is installed, as LiteLLMModel uses it for retrying logic
try:
    import tenacity
except Exception as e:
    raise ImportError(
        "LiteLLMModel requires the 'tenacity' package for retrying logic. "
        "Please install it with 'pip install tenacity'."
    ) from e