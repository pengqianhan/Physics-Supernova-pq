# Information
This project uses an LLM agent to learn hybrid automata from trace data and plot data.

## Key Scripts
- `run_llm_ha_gamma.py`: Main entry point for running the LLM agent pipeline.
- `run.sh`: Example script for running the pipeline on the `ball` dataset to verify that everything works.
- `run_all.sh`: Script for running the pipeline on all datasets to generate full results.
- `test_imagetool.py`: Script for testing the image tool component of the pipeline.
- `test_kimi.py`: Script for testing the Kimi vision model with the liteLLM interface.

## Guidelines
- Keep the pipeline compatible with any LLM API.
- Use `utils/llm_providers.py` as the interface module for different LLM providers. At present, `utils/imgTools_ha.py` can already adapt to different LLM APIs. It uses `genai.Client` only when the input model is Gemini (image processing uses `code_execution=types.ToolCodeExecution`, so `HybridAutomatonImageTool` uses Gemini's `genai.Client` for image handling). Other models all use the LiteLLM interface.
- `run_llm_ha_gamma.py` also needs to be modified in the part that calls the LLM provider so it can adapt to different LLM APIs. Make the parameters `--manager-model`, `summary-model`, `managed-agents-list-model`, and `structured-output-model` compatible with different LLM APIs.