"""
Pure Python Class-Based Hybrid Automaton Learning (SR-Scientist Style)

Completely removes JSON dependency:
- LLM generates Python class directly
- Optimizer refines parameters
- Evaluation uses Python class directly
- No JSON conversion at any step

Usage:
    python run_pure_python_ha.py \\
        --input-data-path data_all/non_linear/duffing \\
        --manager-model gemini/gemini-flash-lite-latest \\
        --max-iterations 3 \\
        --optimization-iters 100
"""

import os
import argparse
import numpy as np
import inspect
from typing import List

# Environment setup
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✓ Environment variables loaded")
except ImportError:
    pass

# smolagents imports
from smolagents import CodeAgent, LiteLLMModel

# Local imports
from utils.pure_python_workflow import generate_pure_python_task, IterationFeedback
from utils.evaluate_pure_python_ha import evaluate_python_ha_class
from utils.utils import IterationResult, ResultsAggregator
from utils.ha_class_validator import (
    validate_python_class_syntax,
    extract_structured_result
)
from utils.llm_evaluator import generate_llm_critique
from utils.imgTools_ha import HybridAutomatonImageTool
from utils.validateTools_ha import ValidateHASpecTool


def get_data_dimensions(input_data_path: str) -> tuple:
    """Auto-detect number of variables and inputs from NPZ file."""
    npz_files = [f for f in os.listdir(input_data_path) if f.endswith('.npz')]
    if not npz_files:
        raise ValueError(f"No .npz files found in {input_data_path}")

    data = np.load(os.path.join(input_data_path, npz_files[0]), allow_pickle=True)

    num_vars = data['state'].shape[0]
    num_inputs = data['input'].shape[0] if 'input' in data and data['input'].ndim > 1 else (1 if 'input' in data else 0)

    print(f"Auto-detected: {num_vars} variable(s), {num_inputs} input(s)")
    return num_vars, num_inputs


def create_agent(model_id: str, tools_list: List[str] = None):
    """Create CodeAgent with specified tools."""
    # Initialize model
    model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200
    )

    # Set up tools
    available_tools = []
    tool_instances = {}

    if tools_list:
        if "hybrid_automaton_image_analysis" in tools_list:
            img_tool = HybridAutomatonImageTool(
                worker_agent=None,
                vision_model_id="gemini/gemini-2.5-flash-lite"
            )
            available_tools.append(img_tool)
            tool_instances["hybrid_automaton_image_analysis"] = img_tool

        if "validate_hybrid_automaton_specification" in tools_list:
            val_tool = ValidateHASpecTool()
            available_tools.append(val_tool)
            tool_instances["validate_hybrid_automaton_specification"] = val_tool

    # Create agent
    agent = CodeAgent(
        tools=available_tools,
        model=model,
        max_steps=100,
        verbosity_level=2
    )

    # Inject worker_agent reference into tools
    for _, tool_instance in tool_instances.items():
        if hasattr(tool_instance, 'worker_agent'):
            tool_instance.worker_agent = agent

    return agent


def main():
    parser = argparse.ArgumentParser(description="Pure Python Class-Based HA Learning")

    parser.add_argument("--input-data-path", type=str, default="data_all/non_linear/duffing",
                        help="Path to trace data directory")
    parser.add_argument("--manager-model", type=str, default="gemini/gemini-flash-lite-latest",
                        help="LLM model ID")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum refinement iterations")
    parser.add_argument("--optimization-iters", type=int, default=50,
                        help="Parameter optimization iterations")
    parser.add_argument("--optimizer-type", type=str, default="simulated_annealing",
                        choices=["simulated_annealing", "hill_climbing"])
    parser.add_argument("--tools-list", nargs="*",
                        default=["hybrid_automaton_image_analysis", "validate_hybrid_automaton_specification"],
                        help="Tools to provide to agent")
    parser.add_argument("--target-error", type=float, default=0.01,
                        help="Target error for early stopping")
    parser.add_argument("--feedback-top-k", type=int, default=3,
                        help="Number of top results to include in feedback")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("PURE PYTHON CLASS-BASED HA LEARNING")
    print("="*80)
    print(f"Model: {args.manager_model}")
    print(f"Data: {args.input_data_path}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Optimization: {args.optimization_iters} iters using {args.optimizer_type}")
    print("="*80 + "\n")

    # Auto-detect dimensions
    num_vars, num_inputs = get_data_dimensions(args.input_data_path)

    # Create agent
    print("Creating agent...")
    agent = create_agent(args.manager_model, args.tools_list)
    print(f"✓ Agent created with {len(agent.tools)} tool(s)\n")

    # Initialize results aggregator
    aggregator = ResultsAggregator(top_k=args.feedback_top_k, min_gap=0.005)

    # Main iteration loop
    for iteration in range(1, args.max_iterations + 1):
        print("\n" + "#"*80)
        print(f"ITERATION {iteration}/{args.max_iterations}")
        print("#"*80 + "\n")

        # Generate structured feedback from previous iterations
        if iteration == 1:
            feedback_list = None
        else:
            # Build IterationFeedback list from ALL previous results
            # Include both successful and failed iterations for learning
            feedback_list = []
            included_iterations = set()

            # First, add top-k successful results (ranked by error)
            for res in aggregator.get_top_k_results():
                feedback_list.append(IterationFeedback(
                    iteration=res.iteration,
                    analysis_process=res.analysis_process or "",
                    class_code=res.class_code or "",
                    metrics=res.metrics,
                    llm_critique=res.llm_critique or "",
                    error_value=res.error_value
                ))
                included_iterations.add(res.iteration)

            # Then, add failed iterations (valuable for learning what NOT to do)
            for res in aggregator.results:
                if res.iteration not in included_iterations:
                    # Include failed results with their error feedback
                    error_critique = res.llm_critique or res.feedback or "(Failed iteration)"
                    feedback_list.append(IterationFeedback(
                        iteration=res.iteration,
                        analysis_process=res.analysis_process or "",
                        class_code=res.class_code or "",
                        metrics=res.metrics,
                        llm_critique=f"[FAILED] {error_critique}",
                        error_value=res.error_value if res.error_value < float('inf') else -1.0  # Mark as failed
                    ))
                    included_iterations.add(res.iteration)

            # Sort by iteration number for chronological order
            feedback_list.sort(key=lambda x: x.iteration)

        # Generate task
        print(f"[{iteration}/4] Generating task...")
        if feedback_list:
            print(f"  Including feedback from {len(feedback_list)} previous iteration(s)")
            for fb in feedback_list:
                status = "FAILED" if fb.error_value < 0 or fb.llm_critique.startswith("[FAILED]") else "OK"
                error_str = "N/A" if fb.error_value < 0 else f"{fb.error_value:.6f}"
                print(f"    - Iter {fb.iteration} [{status}]: error={error_str}, analysis={len(fb.analysis_process)} chars")
        else:
            print(f"  No feedback available (first iteration)")

        task, images = generate_pure_python_task(
            input_data_path=args.input_data_path,
            num_variables=num_vars,
            num_inputs=num_inputs,
            iteration=iteration,
            feedback=feedback_list,
            tools_list=args.tools_list,
            manager_type="CodeAgent"
        )

        # Save task prompt
        task_dir = os.path.join("task_prompts", f"iter_{iteration}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "task.md"), "w") as f:
            f.write(task)

        # Debug: verify feedback is in task
        has_feedback_section = "## Previous Iterations" in task
        print(f"✓ Task generated ({len(task)} chars, feedback_section={has_feedback_section})")

        # Run agent
        print(f"\n[{iteration}/4] Running agent...")
        try:
            result = agent.run(task, images=images)
            print("✓ Agent completed")
        except Exception as e:
            print(f"✗ Agent failed: {e}")
            # Record failed iteration
            aggregator.add_result(IterationResult(
                iteration=iteration,
                ha_specification=None,
                metrics={},
                feedback=f"Agent failed: {str(e)}",
                success=False,
                error_value=float('inf')
            ))
            continue

        # Extract structured result (analysis_process + class_code)
        print(f"\n[{iteration}/4] Extracting structured result...")

        result_str = str(result)
        print(f"  Debug: result type = {type(result)}, result = {result_str[:100]}...")

        # Use structured extraction
        structured = extract_structured_result(result_str)
        class_code = structured.class_code
        analysis_process = structured.analysis_process

        if class_code:
            print("  ✓ Extracted class_code from result")
        if analysis_process:
            print(f"  ✓ Extracted analysis_process ({len(analysis_process)} chars)")

        # Fallback 1: Try extracting from agent logs if no class_code
        if not class_code and hasattr(agent, 'logs'):
            print(f"  Trying fallback: extracting from agent logs ({len(agent.logs)} logs)...")
            for log in reversed(agent.logs):
                log_str = str(log)
                log_structured = extract_structured_result(log_str)
                if log_structured.class_code:
                    class_code = log_structured.class_code
                    if not analysis_process and log_structured.analysis_process:
                        analysis_process = log_structured.analysis_process
                    print("  ✓ Found code in log content")
                    break

                # Check LLM output
                if hasattr(log, 'llm_output') and log.llm_output:
                    llm_structured = extract_structured_result(str(log.llm_output))
                    if llm_structured.class_code:
                        class_code = llm_structured.class_code
                        if not analysis_process and llm_structured.analysis_process:
                            analysis_process = llm_structured.analysis_process
                        print("  ✓ Found code in LLM output")
                        break

                # Check tool input (code execution)
                if hasattr(log, 'tool_calls') and log.tool_calls:
                    for tool_call in log.tool_calls:
                        if hasattr(tool_call, 'arguments'):
                            args_str = str(tool_call.arguments)
                            tool_structured = extract_structured_result(args_str)
                            if tool_structured.class_code:
                                class_code = tool_structured.class_code
                                print("  ✓ Found code in tool call arguments")
                                break
                    if class_code:
                        break

        # Fallback 2: Try extracting from agent's Python executor state
        if not class_code and hasattr(agent, 'python_executor'):
            print("  Trying fallback 2: extracting from executor state...")
            if hasattr(agent.python_executor, '_globals'):
                if 'HybridAutomaton' in agent.python_executor._globals:
                    try:
                        class_obj = agent.python_executor._globals['HybridAutomaton']
                        class_code = inspect.getsource(class_obj)
                        print("  ✓ Extracted source using inspect.getsource()")
                    except (OSError, TypeError):
                        pass

        if not class_code:
            print("✗ Failed to extract Python class from agent output or logs")
            aggregator.add_result(IterationResult(
                iteration=iteration,
                ha_specification=None,
                metrics={},
                feedback="Could not extract Python class from output",
                success=False,
                error_value=float('inf'),
                analysis_process=analysis_process,
                llm_critique="(Extraction failed - no class code found)"
            ))
            continue

        # Validate syntax
        is_valid, errors = validate_python_class_syntax(class_code)
        if not is_valid:
            print(f"✗ Class validation failed: {errors}")
            aggregator.add_result(IterationResult(
                iteration=iteration,
                ha_specification=None,
                metrics={},
                feedback=f"Syntax errors: {errors}",
                success=False,
                error_value=float('inf'),
                class_code=class_code,
                analysis_process=analysis_process,
                llm_critique=f"(Syntax validation failed: {errors})"
            ))
            continue

        print("✓ Python class extracted and validated")
        if analysis_process:
            print(f"  Analysis ({len(analysis_process)} chars): {analysis_process[:100]}...")

        # Evaluate
        print(f"\n[{iteration}/4] Evaluating...")
        success, metrics, feedback_str, opt_class, opt_params = evaluate_python_ha_class(
            class_code=class_code,
            input_data_path=args.input_data_path,
            output_dir=os.path.join("evaluation_results", f"iter_{iteration}"),
            iteration=iteration,
            optimization_iters=args.optimization_iters,
            optimizer_type=args.optimizer_type
        )

        # Extract error value
        error_val = metrics.get('max_diff', float('inf')) if success else float('inf')

        print(f"\n{'✓' if success else '✗'} Evaluation {'succeeded' if success else 'failed'}")
        print(f"Error: {error_val:.6f}")

        # Generate LLM critique
        print(f"\n[{iteration}/4] Generating LLM critique...")
        if success:
            llm_critique = generate_llm_critique(
                class_code=opt_class,
                metrics=metrics,
                analysis_process=analysis_process,
                model_id=args.manager_model
            )
            print(f"✓ Critique generated ({len(llm_critique)} chars)")
        else:
            llm_critique = f"Evaluation failed: {feedback_str}"
            print(f"  (Skipped critique - evaluation failed)")

        # Store result with all structured fields
        aggregator.add_result(IterationResult(
            iteration=iteration,
            ha_specification=None,  # No JSON in pure Python workflow
            metrics=metrics,
            feedback=feedback_str,
            success=success,
            error_value=error_val,
            class_code=opt_class,
            optimized_params=opt_params,
            analysis_process=analysis_process,
            llm_critique=llm_critique
        ))

        # Check early stopping
        should_stop, reason = aggregator.should_early_stop(
            target_error=args.target_error,
            min_iterations=2,
            no_improvement_patience=3
        )

        if should_stop:
            print(f"\n🛑 Early stop: {reason}")
            break

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

    if aggregator.best_result:
        print(f"\n✓ Best error: {aggregator.best_error:.6f} (Iteration {aggregator.best_result.iteration})")

        # Save best result
        best_dir = "evaluation_results/best"
        os.makedirs(best_dir, exist_ok=True)

        if aggregator.best_result.class_code:
            with open(os.path.join(best_dir, "best_ha_class.py"), "w") as f:
                f.write(aggregator.best_result.class_code)
            print(f"Best class saved to: {best_dir}/best_ha_class.py")

        if aggregator.best_result.optimized_params is not None:
            np.save(os.path.join(best_dir, "best_params.npy"), aggregator.best_result.optimized_params)
            print(f"Best params saved to: {best_dir}/best_params.npy")

    else:
        print("\n✗ No successful results")

    print("\nIteration summary:")
    for res in aggregator.results:
        status = "✓" if res.success else "✗"
        error_str = f"{res.error_value:.6f}" if res.error_value < float('inf') else "FAILED"
        best_mark = " (BEST)" if res == aggregator.best_result else ""
        print(f"  Iter {res.iteration}: [{status}] Error={error_str}{best_mark}")


if __name__ == "__main__":
    main()
