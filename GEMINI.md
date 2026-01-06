# Project Context: HA-Scientist / Physics Supernova

## Overview
This project, **HA-Scientist**, is an agentic workflow designed to autonomously identify Hybrid Automaton (HA) models from time-series trajectory data. It is built upon the **Physics Supernova** architecture, an AI agent system originally designed for solving elite-level physics problems (IPhO).

The core philosophy is a **"Scientist-Critic" loop**:
1.  **Scientist:** Proposes mathematical models (HA specifications) to explain observed data.
2.  **Critic:** Evaluates these models against ground truth data using visual and numerical analysis.

## Architecture

The system utilizes the **smolagents** framework with a multi-agent setup:

*   **Manager Agent:** A `CodeAgent` that orchestrates the workflow, plans reasoning, and delegates tasks.
*   **Tools:** Specialized Python classes (inheriting from `smolagents.Tool`) for specific capabilities:
    *   `HybridAutomatonImageTool`: Visual analysis of plots to detect mode switches.
    *   `WolframAlphaTool`: Mathematical computation and unit conversion.
    *   `ReviewRequestTool`: Post-hoc answer checking.
    *   `SummarizeMemoryTool`: Managing long context.
*   **Managed Agents:** Sub-agents for heavy computation (e.g., `data_analysis_expert`).

## Key Directories & Files

*   **`run_llm_ha_beta.py`**: The main entry point for the Hybrid Automaton identification workflow.
*   **`utils/Dainarx_code/`**: The core logic for Hybrid Automata learning and evaluation.
    *   `src/HybridAutomata.py`: Simulation engine.
    *   `src/ODE_System.py`: ODE integration.
    *   `automata/`: JSON specifications of various systems (e.g., Duffing oscillator).
    *   `HA_evaluation.py`: Framework for evaluating learned models against ground truth.
*   **`evaluation_results/`**: Output directory for results, including visual overlays and error metrics.
*   **`prompts_ha/`**: Contains the prompt templates used to guide the LLM agents.
*   **`CLAUDE.md`**: Extensive developer documentation and architectural details (highly recommended for deep dives).

## Development & Usage

### Environment
*   **Language:** Python 3.10+
*   **Key Dependencies:** `smolagents`, `tenacity`, `numpy`, `scipy`, `matplotlib`, `loguru`.
*   **API Keys:** Requires `.env` file with keys like `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `WOLFRAM_APP_ID`.

### Running the Agent
The standard command to run the HA identification process:

```bash
python run_llm_ha_beta.py \
  --input-data-path "utils/Dainarx_code/data_duffing" \
  --manager-model "gemini/gemini-flash-lite-latest" \
  --max-iterations 3 \
  --system-name "Duffing Oscillator"
```

### Data Formats
*   **Trajectory Data (`.npz`):** Contains `state`, `input`, `mode`, and `change_points`.
*   **HA Specification (`.json`):** Defines variables, inputs, modes (ODEs), and edges (transitions/guards). See `utils/Dainarx_code/automata/json_readme.md`.

## Coding Conventions
*   **Agent Pattern:** Tools often use "delayed injection" to access the parent agent's context (e.g., `self.worker_agent`).
*   **Time Indexing:** In HA simulation, `input_array[i]` typically corresponds to time `(i+1)*dt`.
*   **Mode Indexing:** Hybrid Automaton modes are 1-indexed.
