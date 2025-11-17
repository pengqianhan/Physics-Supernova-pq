# Repository Guidelines

## Project Structure & Module Organization
Core agent entry points live in `run.py` and `run_gemini.py`, which coordinate Physics Supernova’s tool-enabled reasoning flow. Shared tools (WolframAlpha, image review, memory summarizer) sit in `utils/`, while batch automation scripts are under `run_scripts/` (e.g., `batchrun.py`). Problem statements and sample outputs are in `examples/Problems/` and `examples/Answers/`. Evaluation helpers reside in `judge_answer/`, and visual assets plus research notes live in `figures/` and `learning_notes/`.

## Build, Test, and Development Commands
Use Python 3.10+ and a virtual environment. Install deps listed in the README:
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install tenacity smolagents smolagents[litellm] loguru python-dotenv
```
Run a single problem with:
```bash
python run.py --input-markdown-file examples/Problems/example/example1problem.md --manager-model openrouter/google/gemini-2.5-pro --manager-type CodeAgent --tools-list "wolfram_alpha_query ask_image_expert ask_review_expert finalize_part_answer"
```
Batch jobs:
```bash
python run_scripts/batchrun.py
```

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indents, snake_case modules/functions, and CapWords classes. Use docstrings for public tool interfaces and type hints when extending agent logic. Prefer explicit logging via `loguru` and keep configuration in `.env` rather than hard-coded strings. Place reusable helpers in `utils/` and keep problem-specific assets inside `examples/`.

## Testing Guidelines
Automated regression currently relies on answer judging. After generating agent outputs, compare them to official solutions with:
```bash
python judge_answer/judge_answers.py --model_dir output/my_run --official_dir examples/Answers --output 0_judged_result.json --max_workers 4
```
Ensure new scripts write results to a dedicated subdirectory inside `output/` and fail gracefully when API keys are missing. Document any manual verification steps in the corresponding run script or README section.

## Commit & Pull Request Guidelines
Recent commits use concise, imperative-style subjects (e.g., “Add method pipeline diagram ...”). Mirror that tone, keep subjects under ~72 chars, and elaborate context in the body if needed. For PRs, include: summary of changes, key commands executed (`run.py`, `batchrun.py`, or judge scripts), any new dependencies, and links to tracked issues. Add screenshots only when UI or figure outputs change.

## Security & Configuration Tips
Store sensitive credentials in `.env` (see `README.md` for required keys) and exclude them from commits. When sharing logs, scrub tokens and problem statements that may be under embargo. Rotate API keys if they are exposed in experiment artifacts.
