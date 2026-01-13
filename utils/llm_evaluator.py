"""
LLM-based evaluator for Hybrid Automaton class code.

Generates critique based on metrics and code analysis.
Uses the same model as the manager agent for consistency.
"""

import os
from typing import Optional


def create_critique_prompt(
    class_code: str,
    metrics: dict,
    analysis_process: str = ""
) -> str:
    """
    Generate prompt for LLM critique.

    Args:
        class_code: Python class code to evaluate
        metrics: Evaluation metrics dict
        analysis_process: Original analysis from generating agent

    Returns:
        Formatted prompt string
    """
    # Format metrics
    metrics_lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            metrics_lines.append(f"- {k}: {v:.6f}")
        else:
            metrics_lines.append(f"- {k}: {v}")
    metrics_str = "\n".join(metrics_lines) if metrics_lines else "(No metrics available)"

    # Truncate code if too long
    code_display = class_code[:2000] + "\n# ... (truncated)" if len(class_code) > 2000 else class_code

    return f"""You are an expert in Hybrid Automaton system identification.

## Task
Evaluate the following Hybrid Automaton Python class and provide constructive feedback.

## Original Analysis (by the generating agent)
{analysis_process if analysis_process else "(Not provided)"}

## Python Class Code
```python
{code_display}
```

## Evaluation Metrics
{metrics_str}

## Your Critique (~100-150 words)
Provide specific, actionable feedback on:
1. **Dynamics accuracy**: Are the ODE equations likely correct based on metrics?
2. **Mode structure**: Is the number of modes appropriate?
3. **Parameter choices**: Are initial parameter values reasonable?
4. **Improvement suggestions**: What specific changes could reduce error?

Focus on actionable improvements. Be concise and specific. Do NOT repeat the code or metrics.
"""


def generate_llm_critique(
    class_code: str,
    metrics: dict,
    analysis_process: str = "",
    model_id: Optional[str] = None
) -> str:
    """
    Generate LLM-based critique of HA class code.

    Args:
        class_code: Python class code
        metrics: Evaluation metrics dict (max_diff, mean_diff, tc, etc.)
        analysis_process: Original analysis from generating agent
        model_id: LLM model to use for critique (same as manager model)

    Returns:
        Critique string (~100-150 words)
    """
    if not model_id:
        model_id = "gemini/gemini-flash-lite-latest"

    try:
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id=model_id,
            api_key=os.environ.get("GEMINI_API_KEY"),
            max_completion_tokens=1024,
            num_retries=2,
            timeout=60
        )

        prompt = create_critique_prompt(class_code, metrics, analysis_process)

        # Use model directly for simple completion
        messages = [{"role": "user", "content": prompt}]
        response = model(messages)

        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    except Exception as e:
        return f"(Critique generation failed: {e})"


# Test block
if __name__ == "__main__":
    print("Testing LLM Evaluator")
    print("=" * 80)

    # Test prompt generation
    test_code = """class HybridAutomaton:
    def __init__(self):
        self.params = [-0.1, -1.0, -1.0, 1.0, 0, 0, 0, 0, 0, 0]
        self.var = "x1"
        self.input = "u1"
        self.order = 2

    def num_modes(self):
        return 1

    def mode_dynamics(self, mode_id, x, u):
        if mode_id == 1:
            return f"x1[2] = {self.params[0]}*x1[1] + {self.params[1]}*x1[0] + {self.params[2]}*x1[0]**3 + {self.params[3]}*u['u1']"
        raise ValueError(f"Unknown mode: {mode_id}")

    def guard_condition(self, source_mode, target_mode, x, u):
        return False

    def reset_map(self, source_mode, target_mode, x, u):
        pass
"""

    test_metrics = {
        "max_diff": 0.0523,
        "mean_diff": 0.0124,
        "tc": 0.0,
        "rmse": 0.0189
    }

    test_analysis = "I observed oscillatory behavior with decaying amplitude, suggesting a damped nonlinear oscillator. The cubic nonlinearity indicates a Duffing-like system."

    print("\n--- Test Prompt ---")
    prompt = create_critique_prompt(test_code, test_metrics, test_analysis)
    print(prompt[:500] + "...")

    print("\n--- Test Critique Generation ---")
    # Only run actual critique if API key is available
    if os.environ.get("GEMINI_API_KEY"):
        critique = generate_llm_critique(
            test_code, test_metrics, test_analysis,
            model_id="gemini/gemini-flash-lite-latest"
        )
        print(critique)
    else:
        print("(Skipping actual API call - no GEMINI_API_KEY set)")

    print("\n" + "=" * 80)
    print("Test complete")
