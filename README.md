# HA-Scientist: Hybrid Automaton System Identification Agent 🔬

**HA-Scientist** is an agentic workflow designed to autonomously identify Hybrid Automaton (HA) models from time-series trajectory data. It iteratively refines mathematical models to capture the switching dynamics of unknown physical systems.

## Key Features ✨

- **"Scientist-Critic" Loop**: An iterative process where the agent acts as a scientist proposing models and a critic evaluating them against ground truth data.
- **Visual & Numerical Fusion**:
  - **Visual Analysis**: Uses `hybrid_automaton_image_analysis` to detect mode switches and non-smooth behaviors from plots.
  - **Numerical Verification**: Uses managed agents (e.g., `data_analysis_expert`) for curve fitting and residual analysis.
- **Automated Validation**: Ensures all generated HA specifications are syntactically valid and physically consistent.
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

The main entry point is `run_llm_ha_gamma.py`.

### Basic Command
```bash
bash run.sh
```

### Output 📊
Results are saved in `evaluation_results/`:

---

## Code Reference: Physics Supernova ⚙️

This project builds upon the architecture of [Physics Supernova](https://github.com/CharlesQ9/Physics-Supernova), an agent system capable of solving elite-level Physics problems.

### Acknowledgments 🙏
Built on [`smolagents`](https://github.com/huggingface/smolagents).

### Phoenix Traces 
To start the Phoenix server, run the following command:
```bash
python -m phoenix.server.main serve
```