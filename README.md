# HA-Scientist: Hybrid Automaton System Identification Agent 🔬

**HA-Scientist** is an agentic workflow designed to autonomously identify Hybrid Automaton (HA) models from time-series trajectory data. It iteratively refines mathematical models to capture the switching dynamics of unknown physical systems.

## Key Features ✨

- **"Scientist-Critic" Loop**: An iterative process where the agent acts as a scientist proposing models and a critic evaluating them against ground truth data.
- **Visual & Numerical Fusion**:
  - **Visual Analysis**: Uses `hybrid_automaton_image_analysis` to detect mode switches and non-smooth behaviors from plots.
  - **Numerical Verification**: Uses managed agents (e.g., `data_analysis_expert`) for curve fitting and residual analysis.
- **Automated Validation**: Ensures all generated HA specifications are syntactically valid and physically consistent.
- **Complex System Handling**: Capable of identifying multi-mode systems like the Duffing Oscillator.

---

## Installation 🧰

Requires Python 3.10+.

```bash
python -m pip install -U pip
pip install tenacity smolagents[litellm] loguru python-dotenv
```

### Set up API Keys 🔑
Create a `.env` file:
```bash
GEMINI_API_KEY=...                # Google Gemini API Key
```

---

## Usage 🚀

The main entry point is `run_llm_ha_beta.py`.

### Basic Command
```bash
python run_llm_ha_beta.py \
  --input-data-path "utils/Dainarx_code/data_duffing" \
  --manager-model "gemini/gemini-flash-lite-latest" \
  --max-iterations 3 \
  --system-name "Duffing Oscillator"
```

### Arguments
- `--input-data-path`: Directory containing the `.npz` trace data files.
- `--manager-model`: The LLM model ID for the main agent (e.g., `gemini/gemini-flash-lite-latest`).
- `--max-iterations`: Maximum number of refinement iterations (default: 3).
- `--tools-list`: List of tools to use. Defaults include:
  - `hybrid_automaton_image_analysis`
  - `summarize_hybrid_automaton_iterations`
  - `validate_hybrid_automaton_specification`
- `--managed-agents-list`: Sub-agents for heavy computation (e.g., `data_analysis_expert`).

### Output 📊
Results are saved in `evaluation_results/`:
- **`ha_eval_<timestamp>.png`**: Visual overlay of the identified HA trajectories vs. ground truth.
- **`ha_eval_<timestamp>.txt`**: Detailed error metrics and the final JSON specification of the Hybrid Automaton.

---

## Code Reference: Physics Supernova ⚙️

This project builds upon the architecture of **Physics Supernova**, an agent system capable of solving elite-level Physics problems.

**Physics Supernova Features Included:**
- **CodeAgent Manager**: Orchestrates tools and reasoning.
- **Specialized Physics Tools**: Image analysis, WolframAlpha integration, and Answer Reviewers.

### Citation 🔗

If you find this work useful, please cite the Physics Supernova paper:

```bibtex
@misc{qiu2025physicssupernovaaiagent,
      title={Physics Supernova: AI Agent Matches Elite Gold Medalists at IPhO 2025}, 
      author={Jiahao Qiu and Jingzhe Shi and Xinzhe Juan and Zelin Zhao and Jiayi Geng and Shilong Liu and Hongru Wang and Sanfeng Wu and Mengdi Wang},
      year={2025},
      eprint={2509.01659},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.01659}, 
}
```

### Acknowledgments 🙏
Built on [`smolagents`](https://github.com/huggingface/smolagents).
