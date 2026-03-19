# Information
This project uses an LLM agent to learn hybrid automata from trace data and plot data.

## Key Scripts
- `run_llm_ha_gamma.py`: Main entry point for running the LLM agent pipeline.
- `run.sh`: Example script for running the pipeline on the `ball` dataset to verify that everything works.
- `run_all.sh`: Script for running the pipeline on all datasets to generate full results.

## Guidelines
- Keep the pipeline compatible with any LLM API.
- Use `utils/llm_providers.py` as the interface module for different LLM providers.