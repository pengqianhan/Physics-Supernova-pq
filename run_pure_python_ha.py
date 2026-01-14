"""
Pure Python Class-Based Hybrid Automaton Learning (SR-Scientist Style)

All generated HA classes inherit from HybridAutomatonBase for consistent interface.

Usage:
    python run_pure_python_ha.py \\
        --input-data-path data_all/non_linear/duffing \\
        --manager-model gemini/gemini-flash-lite-latest \\
        --max-iterations 3 \\
        --optimization-iters 100
"""

import os
import json
import argparse
import inspect
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import asdict

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# === Agent Monitoring (Phoenix - Free Local Solution) ===
try:
    from phoenix.otel import register
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    register()
    SmolagentsInstrumentor().instrument()
    print("[Telemetry] Phoenix monitoring enabled - visit http://localhost:6006")
except ImportError:
    print("[Telemetry] Phoenix not installed. Install: pip install arize-phoenix openinference-instrumentation-smolagents")
# =============================================

from smolagents import CodeAgent, LiteLLMModel

from utils.pure_python_workflow import generate_pure_python_task, IterationFeedback
from utils.evaluate_pure_python_ha import evaluate_python_ha_class
from utils.utils import IterationResult, ResultsAggregator
from utils.ha_class_validator import validate_python_class_syntax, extract_structured_result
from utils.llm_evaluator import generate_llm_critique
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.validateTools_ha import ValidateHASpecTool
from utils.ha_base_class import HybridAutomatonBase


def get_data_dimensions(input_data_path: str) -> Tuple[int, int]:
    """Auto-detect number of variables and inputs from NPZ file."""
    npz_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not npz_files:
        raise ValueError(f"No .npz files found in {input_data_path}")

    data = np.load(os.path.join(input_data_path, npz_files[0]), allow_pickle=True)
    num_vars = data['state'].shape[0]
    num_inputs = data['input'].shape[0] if 'input' in data and data['input'].ndim > 1 else (1 if 'input' in data else 0)
    return num_vars, num_inputs


def create_agent(model_id: str, tools_list: Optional[List[str]] = None) -> CodeAgent:
    """Create CodeAgent with specified tools and base class in execution namespace."""
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200
    )

    tool_map = {
        "hybrid_automaton_image_analysis": lambda: HybridAutomatonImageTool(
            worker_agent=None, vision_model_id="gemini/gemini-flash-latest"
        ),
        "validate_hybrid_automaton_specification": ValidateHASpecTool
    }

    tools = []
    if tools_list:
        for name in tools_list:
            if name in tool_map:
                tools.append(tool_map[name]())

    agent = CodeAgent(tools=tools, model=model, max_steps=100, verbosity_level=2)

    # Inject base class into agent's execution namespace
    # This allows the agent to define classes that inherit from HybridAutomatonBase
    if hasattr(agent, 'python_executor') and hasattr(agent.python_executor, 'state'):
        agent.python_executor.state['HybridAutomatonBase'] = HybridAutomatonBase

    for tool in agent.tools.values():
        if hasattr(tool, 'worker_agent'):
            tool.worker_agent = agent

    return agent


def extract_class_from_agent(result: str, agent) -> Tuple[str, str]:
    """Extract class code and analysis from agent output, with fallbacks."""
    structured = extract_structured_result(str(result))
    class_code = structured.class_code
    analysis = structured.analysis_process

    if class_code:
        return class_code, analysis

    # Fallback: search agent logs
    if hasattr(agent, 'logs'):
        for log in reversed(agent.logs):
            for source in [str(log), getattr(log, 'llm_output', None)]:
                if source:
                    s = extract_structured_result(str(source))
                    if s.class_code:
                        return s.class_code, analysis or s.analysis_process

            if hasattr(log, 'tool_calls') and log.tool_calls:
                for tc in log.tool_calls:
                    if hasattr(tc, 'arguments'):
                        s = extract_structured_result(str(tc.arguments))
                        if s.class_code:
                            return s.class_code, analysis

    # Fallback: extract from executor state
    if hasattr(agent, 'python_executor') and hasattr(agent.python_executor, '_globals'):
        if 'HybridAutomaton' in agent.python_executor._globals:
            try:
                return inspect.getsource(agent.python_executor._globals['HybridAutomaton']), analysis
            except (OSError, TypeError):
                pass

    return "", analysis


def build_feedback_list(aggregator: ResultsAggregator) -> List[IterationFeedback]:
    """Build feedback list from aggregator results for next iteration."""
    feedback_list = []
    included = set()

    # Add top-k successful results first
    for res in aggregator.get_top_k_results():
        feedback_list.append(IterationFeedback(
            iteration=res.iteration,
            analysis_process=res.analysis_process or "",
            class_code=res.class_code or "",
            metrics=res.metrics,
            llm_critique=res.llm_critique or "",
            error_value=res.error_value,
            plot_path=res.plot_path or ""
        ))
        included.add(res.iteration)

    # Add failed iterations for learning
    for res in aggregator.results:
        if res.iteration not in included:
            critique = res.llm_critique or res.feedback or "(Failed iteration)"
            feedback_list.append(IterationFeedback(
                iteration=res.iteration,
                analysis_process=res.analysis_process or "",
                class_code=res.class_code or "",
                metrics=res.metrics,
                llm_critique=f"[FAILED] {critique}",
                error_value=-1.0 if res.error_value >= float('inf') else res.error_value,
                plot_path=res.plot_path or ""
            ))

    feedback_list.sort(key=lambda x: x.iteration)
    return feedback_list


def save_reasoning_log(
    log_dir: str,
    iteration: int,
    task_prompt: str,
    agent,
    result: Any,
    class_code: str,
    analysis: str,
    metrics: Dict,
    llm_critique: str,
    append_to_master: bool = True
) -> str:
    """Save complete reasoning trace for an iteration.

    Returns the path to the saved log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Extract memory steps from agent
    memory_steps = []
    if hasattr(agent, 'memory') and agent.memory:
        for step in agent.memory.steps:
            step_dict = {}
            step_dict['type'] = type(step).__name__

            # Convert dataclass to dict, handling non-serializable objects
            try:
                raw_dict = asdict(step)
                # Clean up non-JSON-serializable items
                for k, v in raw_dict.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        step_dict[k] = v
                    elif isinstance(v, (list, tuple)):
                        step_dict[k] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in v]
                    elif isinstance(v, dict):
                        step_dict[k] = {str(kk): str(vv) if not isinstance(vv, (str, int, float, bool, type(None))) else vv for kk, vv in v.items()}
                    else:
                        step_dict[k] = str(v)
            except Exception:
                # Fallback: just get string representations of attributes
                for attr in ['llm_output', 'tool_calls', 'observations', 'error', 'model_output']:
                    if hasattr(step, attr):
                        val = getattr(step, attr)
                        step_dict[attr] = str(val) if val is not None else None

            memory_steps.append(step_dict)

    # Build the complete log entry
    log_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "input_prompt": task_prompt,
        "reasoning_steps": memory_steps,
        "final_result": str(result) if result else None,
        "extracted_class_code": class_code,
        "analysis_process": analysis,
        "evaluation_metrics": metrics,
        "llm_critique": llm_critique
    }

    # Save individual iteration log (JSON)
    iter_log_path = os.path.join(log_dir, f"iter_{iteration}_reasoning.json")
    with open(iter_log_path, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    # Also save a human-readable markdown version
    md_path = os.path.join(log_dir, f"iter_{iteration}_reasoning.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Iteration {iteration} Reasoning Trace\n\n")
        f.write(f"**Timestamp:** {log_entry['timestamp']}\n\n")

        f.write("## Input Prompt\n\n")
        f.write("```\n")
        f.write(task_prompt[:5000] + ("..." if len(task_prompt) > 5000 else ""))
        f.write("\n```\n\n")

        f.write("## Reasoning Steps\n\n")
        for i, step in enumerate(memory_steps, 1):
            f.write(f"### Step {i} ({step.get('type', 'Unknown')})\n\n")
            if step.get('llm_output'):
                f.write("**LLM Output:**\n```\n")
                output = str(step['llm_output'])[:3000]
                f.write(output + ("..." if len(str(step.get('llm_output', ''))) > 3000 else ""))
                f.write("\n```\n\n")
            if step.get('tool_calls'):
                f.write(f"**Tool Calls:** {step['tool_calls']}\n\n")
            if step.get('observations'):
                f.write("**Observations:**\n```\n")
                obs = str(step['observations'])[:2000]
                f.write(obs + ("..." if len(str(step.get('observations', ''))) > 2000 else ""))
                f.write("\n```\n\n")
            if step.get('error'):
                f.write(f"**Error:** {step['error']}\n\n")

        f.write("## Extracted Class Code\n\n")
        f.write("```python\n")
        f.write(class_code if class_code else "(None extracted)")
        f.write("\n```\n\n")

        f.write("## Evaluation Metrics\n\n")
        f.write(f"```json\n{json.dumps(metrics, indent=2)}\n```\n\n")

        f.write("## LLM Critique\n\n")
        f.write(llm_critique if llm_critique else "(No critique)")
        f.write("\n")

    # Append to master log file for all iterations
    if append_to_master:
        master_log_path = os.path.join(log_dir, "master_reasoning_log.json")
        master_data = []
        if os.path.exists(master_log_path):
            try:
                with open(master_log_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)
            except json.JSONDecodeError:
                master_data = []

        # Replace or append this iteration
        master_data = [e for e in master_data if e.get('iteration') != iteration]
        master_data.append(log_entry)
        master_data.sort(key=lambda x: x.get('iteration', 0))

        with open(master_log_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)

    print(f"  [Log] Saved reasoning trace to {iter_log_path}")
    return iter_log_path


def record_failure(aggregator: ResultsAggregator, iteration: int, reason: str,
                   analysis: str = "", class_code: str = "") -> None:
    """Record a failed iteration result."""
    aggregator.add_result(IterationResult(
        iteration=iteration,
        ha_specification=None,
        metrics={},
        feedback=reason,
        success=False,
        error_value=float('inf'),
        analysis_process=analysis,
        class_code=class_code,
        llm_critique=f"({reason})",
        plot_path=None
    ))


def main():
    parser = argparse.ArgumentParser(description="Pure Python Class-Based HA Learning")
    parser.add_argument("--input-data-path", type=str, default="data_all/ATVA/ball")
    parser.add_argument("--manager-model", type=str, default="gemini/gemini-flash-lite-latest")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--optimization-iters", type=int, default=50)
    parser.add_argument("--optimizer-type", type=str, default="simulated_annealing",
                        choices=["simulated_annealing", "hill_climbing"])
    parser.add_argument("--tools-list", nargs="*",
                        default=["hybrid_automaton_image_analysis", "validate_hybrid_automaton_specification"])
    parser.add_argument("--target-error", type=float, default=0.01)
    parser.add_argument("--feedback-top-k", type=int, default=3)
    parser.add_argument("--save-reasoning-log", type=str, default="reasoning_logs",
                        help="Directory to save reasoning logs (set to empty string to disable)")
    args = parser.parse_args()

    print(f"\n{'='*60}\nPURE PYTHON HA LEARNING\n{'='*60}")
    print(f"Model: {args.manager_model} | Data: {args.input_data_path}")
    print(f"Iterations: {args.max_iterations} | Optimization: {args.optimization_iters} ({args.optimizer_type})")

    num_vars, num_inputs = get_data_dimensions(args.input_data_path)
    print(f"Detected: {num_vars} var(s), {num_inputs} input(s)")

    agent = create_agent(args.manager_model, args.tools_list)
    aggregator = ResultsAggregator(top_k=args.feedback_top_k, min_gap=0.005)

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*60}\nITERATION {iteration}/{args.max_iterations}\n{'#'*60}")

        # Build feedback
        feedback_list = build_feedback_list(aggregator) if iteration > 1 else None
        if feedback_list:
            print(f"Feedback from {len(feedback_list)} previous iteration(s)")

        # Generate and save task
        task, images = generate_pure_python_task(
            input_data_path=args.input_data_path,
            num_variables=num_vars,
            num_inputs=num_inputs,
            iteration=iteration,
            feedback=feedback_list,
            tools_list=args.tools_list,
            manager_type="CodeAgent"
        )

        task_dir = os.path.join("task_prompts", f"iter_{iteration}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "task.md"), "w") as f:
            f.write(task)

        # Run agent
        print(f"[1/4] Running agent...")
        result = None
        try:
            result = agent.run(task, images=images)
        except Exception as e:
            print(f"Agent failed: {e}")
            record_failure(aggregator, iteration, f"Agent failed: {e}")
            if args.save_reasoning_log:
                save_reasoning_log(
                    log_dir=args.save_reasoning_log, iteration=iteration,
                    task_prompt=task, agent=agent, result=None,
                    class_code="", analysis="", metrics={},
                    llm_critique=f"[FAILED] Agent exception: {e}"
                )
            continue

        # Extract class code
        print(f"[2/4] Extracting class...")
        class_code, analysis = extract_class_from_agent(result, agent)

        if not class_code:
            print("Failed to extract class code")
            record_failure(aggregator, iteration, "Could not extract Python class", analysis)
            if args.save_reasoning_log:
                save_reasoning_log(
                    log_dir=args.save_reasoning_log, iteration=iteration,
                    task_prompt=task, agent=agent, result=result,
                    class_code="", analysis=analysis, metrics={},
                    llm_critique="[FAILED] Could not extract Python class from agent output"
                )
            continue

        # Validate syntax
        is_valid, errors = validate_python_class_syntax(class_code)
        if not is_valid:
            print(f"Validation failed: {errors}")
            record_failure(aggregator, iteration, f"Syntax errors: {errors}", analysis, class_code)
            if args.save_reasoning_log:
                save_reasoning_log(
                    log_dir=args.save_reasoning_log, iteration=iteration,
                    task_prompt=task, agent=agent, result=result,
                    class_code=class_code, analysis=analysis, metrics={},
                    llm_critique=f"[FAILED] Syntax validation errors: {errors}"
                )
            continue

        print(f"Class extracted ({len(class_code)} chars)")

        # Evaluate
        print(f"[3/4] Evaluating...")
        success, metrics, feedback_str, opt_class, opt_params, plot_path = evaluate_python_ha_class(
            class_code=class_code,
            input_data_path=args.input_data_path,
            output_dir=os.path.join("evaluation_results", f"iter_{iteration}"),
            iteration=iteration,
            optimization_iters=args.optimization_iters,
            optimizer_type=args.optimizer_type
        )

        error_val = metrics.get('max_diff', float('inf')) if success else float('inf')
        print(f"{'Success' if success else 'Failed'}: error={error_val:.6f}")

        # Generate critique
        print(f"[4/4] Generating critique...")
        llm_critique = (
            generate_llm_critique(opt_class, metrics, analysis, args.manager_model)
            if success else f"Evaluation failed: {feedback_str}"
        )

        aggregator.add_result(IterationResult(
            iteration=iteration,
            ha_specification=None,
            metrics=metrics,
            feedback=feedback_str,
            success=success,
            error_value=error_val,
            class_code=opt_class,
            optimized_params=opt_params,
            analysis_process=analysis,
            llm_critique=llm_critique,
            plot_path=plot_path
        ))

        # Save reasoning log if enabled
        if args.save_reasoning_log:
            save_reasoning_log(
                log_dir=args.save_reasoning_log,
                iteration=iteration,
                task_prompt=task,
                agent=agent,
                result=result,
                class_code=opt_class or class_code,
                analysis=analysis,
                metrics=metrics,
                llm_critique=llm_critique
            )

        # Early stopping check
        should_stop, reason = aggregator.should_early_stop(
            target_error=args.target_error, min_iterations=2, no_improvement_patience=3
        )
        if should_stop:
            print(f"\nEarly stop: {reason}")
            break

    # Summary
    print(f"\n{'='*60}\nEXPERIMENT COMPLETE\n{'='*60}")

    if aggregator.best_result:
        print(f"Best error: {aggregator.best_error:.6f} (Iteration {aggregator.best_result.iteration})")

        best_dir = "evaluation_results/best"
        os.makedirs(best_dir, exist_ok=True)

        if aggregator.best_result.class_code:
            with open(os.path.join(best_dir, "best_ha_class.py"), "w") as f:
                f.write(aggregator.best_result.class_code)

        if aggregator.best_result.optimized_params is not None:
            np.save(os.path.join(best_dir, "best_params.npy"), aggregator.best_result.optimized_params)
    else:
        print("No successful results")

    print("\nSummary:")
    for res in aggregator.results:
        mark = " *" if res == aggregator.best_result else ""
        err = f"{res.error_value:.6f}" if res.error_value < float('inf') else "FAILED"
        print(f"  Iter {res.iteration}: {err}{mark}")


if __name__ == "__main__":
    main()
