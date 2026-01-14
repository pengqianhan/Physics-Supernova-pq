"""
Pure Python Class-Based Hybrid Automaton Learning (SR-Scientist Style)

Usage:
    python run_pure_python_ha.py \\
        --input-data-path data_all/non_linear/duffing \\
        --manager-model gemini/gemini-flash-lite-latest \\
        --max-iterations 3 \\
        --optimization-iters 100
"""

import os
import argparse
import inspect
import numpy as np
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# === Agent 运行监控 (Phoenix - 免费本地方案) ===
try:
    from phoenix.otel import register
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    register()
    SmolagentsInstrumentor().instrument()
    print("[Telemetry] Phoenix 监控已启用 - 访问 http://localhost:6006")
except ImportError:
    print("[Telemetry] Phoenix 未安装，跳过监控。安装: pip install arize-phoenix openinference-instrumentation-smolagents")
# =============================================

from smolagents import CodeAgent, LiteLLMModel

from utils.pure_python_workflow import generate_pure_python_task, IterationFeedback
from utils.evaluate_pure_python_ha import evaluate_python_ha_class
from utils.utils import IterationResult, ResultsAggregator
from utils.ha_class_validator import validate_python_class_syntax, extract_structured_result
from utils.llm_evaluator import generate_llm_critique
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.validateTools_ha import ValidateHASpecTool


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
    """Create CodeAgent with specified tools."""
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200
    )

    tool_map = {
        "hybrid_automaton_image_analysis": lambda: HybridAutomatonImageTool(
            worker_agent=None, vision_model_id="gemini/gemini-2.5-flash-lite"
        ),
        "validate_hybrid_automaton_specification": ValidateHASpecTool
    }

    tools = []
    if tools_list:
        for name in tools_list:
            if name in tool_map:
                tools.append(tool_map[name]())

    agent = CodeAgent(tools=tools, model=model, max_steps=100, verbosity_level=2)

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
        try:
            result = agent.run(task, images=images)
        except Exception as e:
            print(f"Agent failed: {e}")
            record_failure(aggregator, iteration, f"Agent failed: {e}")
            continue

        # Extract class code
        print(f"[2/4] Extracting class...")
        class_code, analysis = extract_class_from_agent(result, agent)

        if not class_code:
            print("Failed to extract class code")
            record_failure(aggregator, iteration, "Could not extract Python class", analysis)
            continue

        # Validate syntax
        is_valid, errors = validate_python_class_syntax(class_code)
        if not is_valid:
            print(f"Validation failed: {errors}")
            record_failure(aggregator, iteration, f"Syntax errors: {errors}", analysis, class_code)
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
