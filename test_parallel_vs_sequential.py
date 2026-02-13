"""
Test: Compare optimize_ha_params.py default parallel run vs num_workers=1 (sequential)

Compares:
  1. Default mode:  num_workers = max(1, min(16, cpu_count // 4)),
                    budget = base_budget * sqrt(num_workers)
  2. Sequential:    num_workers = 1, budget = base_budget

Metrics reported:
  - Wall-clock time (seconds)
  - Final optimization error (mean_diff)
  - Per-ground-truth evaluation errors

Usage:
    python test_parallel_vs_sequential.py
    python test_parallel_vs_sequential.py --trials 3        # repeat N times
    python test_parallel_vs_sequential.py --train_num 2     # use 2 GT files
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Any

warnings.filterwarnings("ignore")

import numpy as np

# Import the optimization function
from optimize_ha_params import optimize_ha_parameters_generic

# ── The same HA spec used in optimize_ha_params.py __main__ ──────────
DUFFING_HA_SPEC = {
    "automaton": {
        "var": "x",
        "input": "u",
        "mode": [
            {"id": 1, "eq": "x[2] = u - 0.3 * x[1] + x[0] - 1* x[0] ** 3"},
            {"id": 2, "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1* x[0] ** 3"},
        ],
        "edge": [
            {
                "direction": "1 -> 2",
                "condition": "abs(x) <= 0.9",
                "reset": {"x": ["", "x[1] * 0.3"]},
            },
            {
                "direction": "2 -> 1",
                "condition": "abs(x) >= 1.2",
                "reset": {"x": ["", "x[1] * 0.95"]},
            },
        ],
    },
    "config": {
        "dt": 0.001,
        "total_time": 10.0,
        "order": 2,
        "need_reset": True,
    },
}

INPUT_DATA_PATH = "data_all/non_linear/duffing"


# ── Evaluate on all ground truth files ───────────────────────────────
def evaluate_on_all_gt(
    ha_spec: Dict[str, Any],
    input_data_path: str,
    max_files: int = 15,
) -> Dict[str, Any]:
    """Evaluate optimized HA spec on ALL ground truth files."""
    import re
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils/Dainarx_code"))
    from HA_evaluation import HAEvaluator

    gt_dir = input_data_path.rstrip("/") + "_g"
    gt_files = sorted(
        [f for f in os.listdir(gt_dir) if f.startswith("ground_truth") and f.endswith(".npz")],
        key=lambda f: int(re.search(r"(\d+)", f).group(1)) if re.search(r"(\d+)", f) else 999,
    )[:max_files]

    dt = ha_spec.get("config", {}).get("dt", 0.001)
    total_time = ha_spec.get("config", {}).get("total_time", 10.0)

    errors = []
    for fname in gt_files:
        npz_path = os.path.join(gt_dir, fname)
        try:
            evaluator = HAEvaluator(
                ha_dict=ha_spec, npz_file_path=npz_path, dt=dt, total_time=total_time
            )
            evaluator.load_ground_truth()
            evaluator.simulate()
            metrics = evaluator.compute_metrics()
            md = metrics.get("mean_diff", float("inf"))
            errors.append(md if md is not None else float("inf"))
        except Exception as e:
            errors.append(float("inf"))

    return {
        "per_file_errors": errors,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "num_files": len(errors),
    }


# ── Single run ───────────────────────────────────────────────────────
def run_once(
    num_workers_override,
    train_num: int = 3,
    label: str = "",
) -> Dict[str, Any]:
    """Run optimization with given num_workers and return results + timing."""
    print(f"\n{'='*70}")
    print(f"  RUN: {label}")
    print(f"  num_workers = {num_workers_override}")
    print(f"{'='*70}")

    t0 = time.perf_counter()

    result = optimize_ha_parameters_generic(
        ha_spec=DUFFING_HA_SPEC,
        input_data_path=INPUT_DATA_PATH,
        model_id="gemini/gemini-3-flash-preview",
        verbose=True,
        train_num=train_num,
        num_workers=num_workers_override,
    )

    elapsed = time.perf_counter() - t0

    # Evaluate on ALL ground truth files
    print(f"\n[Eval] Evaluating optimized spec on all ground truth files...")
    eval_results = evaluate_on_all_gt(result["ha_spec"], INPUT_DATA_PATH)

    return {
        "label": label,
        "num_workers": num_workers_override,
        "train_error": result["error"],
        "wall_time_s": elapsed,
        "parameters": result["parameters"],
        "eval_all_gt": eval_results,
    }


# ── Main comparison ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare parallel vs sequential HA optimization")
    parser.add_argument("--trials", type=int, default=3, help="Number of repeated trials (default: 1)")
    parser.add_argument("--train_num", type=int, default=3, help="Number of GT files for training (default: 1)")
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(16, cpu_count // 4))

    print("#" * 70)
    print("#  TEST: Parallel (default) vs Sequential (num_workers=1)")
    print(f"#  CPU count: {cpu_count}")
    print(f"#  Default num_workers: {default_workers}")
    print(f"#  Trials: {args.trials}")
    print(f"#  Train files: {args.train_num}")
    print("#" * 70)

    all_results = {"sequential": [], "parallel": []}

    for trial in range(args.trials):
        print(f"\n\n{'#'*70}")
        print(f"#  TRIAL {trial + 1} / {args.trials}")
        print(f"{'#'*70}")

        # --- Sequential (num_workers=1) ---
        seq_result = run_once(
            num_workers_override=1,
            train_num=args.train_num,
            label=f"Sequential (num_workers=1) [trial {trial+1}]",
        )
        all_results["sequential"].append(seq_result)

        # --- Parallel (default num_workers) ---
        par_result = run_once(
            num_workers_override=None,  # use default
            train_num=args.train_num,
            label=f"Parallel (default num_workers={default_workers}) [trial {trial+1}]",
        )
        all_results["parallel"].append(par_result)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n")
    print("=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)

    # Collect stats
    for mode_key, mode_label in [("sequential", "Sequential (num_workers=1)"),
                                   ("parallel", f"Parallel (default num_workers={default_workers})")]:
        results = all_results[mode_key]
        times = [r["wall_time_s"] for r in results]
        train_errors = [r["train_error"] for r in results]
        eval_means = [r["eval_all_gt"]["mean_error"] for r in results]

        print(f"\n  {mode_label}:")
        print(f"  {'─'*60}")

        if args.trials == 1:
            r = results[0]
            actual_nw = r["num_workers"] if r["num_workers"] is not None else default_workers
            print(f"    num_workers:        {actual_nw}")
            print(f"    Wall-clock time:    {r['wall_time_s']:.2f} s")
            print(f"    Train error:        {r['train_error']:.6f}")
            print(f"    Eval mean error:    {r['eval_all_gt']['mean_error']:.6f}")
            print(f"    Eval median error:  {r['eval_all_gt']['median_error']:.6f}")
            print(f"    Eval min/max error: {r['eval_all_gt']['min_error']:.6f} / {r['eval_all_gt']['max_error']:.6f}")
            print(f"    Optimized params:")
            for name, val in r["parameters"].items():
                print(f"      {name}: {val:.6f}")
        else:
            print(f"    Wall-clock time:  mean={np.mean(times):.2f}s  std={np.std(times):.2f}s  "
                  f"[{', '.join(f'{t:.1f}' for t in times)}]")
            print(f"    Train error:      mean={np.mean(train_errors):.6f}  std={np.std(train_errors):.6f}  "
                  f"[{', '.join(f'{e:.6f}' for e in train_errors)}]")
            print(f"    Eval mean error:  mean={np.mean(eval_means):.6f}  std={np.std(eval_means):.6f}  "
                  f"[{', '.join(f'{e:.6f}' for e in eval_means)}]")

    # ── Head-to-head ─────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  HEAD-TO-HEAD COMPARISON:")
    print(f"  {'─'*60}")

    seq_times = [r["wall_time_s"] for r in all_results["sequential"]]
    par_times = [r["wall_time_s"] for r in all_results["parallel"]]
    seq_eval = [r["eval_all_gt"]["mean_error"] for r in all_results["sequential"]]
    par_eval = [r["eval_all_gt"]["mean_error"] for r in all_results["parallel"]]

    speedup = np.mean(seq_times) / np.mean(par_times) if np.mean(par_times) > 0 else float("inf")
    error_ratio = np.mean(par_eval) / np.mean(seq_eval) if np.mean(seq_eval) > 0 else float("inf")

    print(f"    Speedup (parallel vs sequential):     {speedup:.2f}x")
    print(f"    Error ratio (parallel / sequential):  {error_ratio:.4f}")
    print(f"    Theoretical speedup (√{default_workers}):            {math.sqrt(default_workers):.2f}x")

    if error_ratio <= 1.1:
        verdict = "PASS - Parallel achieves comparable accuracy"
    elif error_ratio <= 1.5:
        verdict = "MARGINAL - Parallel slightly worse accuracy"
    else:
        verdict = "FAIL - Parallel significantly worse accuracy"

    if speedup > 1.2:
        speed_verdict = f"Parallel is {speedup:.1f}x faster"
    elif speedup > 0.8:
        speed_verdict = "Similar wall-clock time"
    else:
        speed_verdict = f"Sequential is {1/speedup:.1f}x faster"

    print(f"\n    Accuracy verdict: {verdict}")
    print(f"    Speed verdict:    {speed_verdict}")
    print()

    # ── Per-GT-file detail table ─────────────────────────────────────
    print("  PER-GROUND-TRUTH FILE ERRORS (last trial):")
    print(f"  {'GT File':<20} {'Sequential':>14} {'Parallel':>14} {'Ratio':>10}")
    print(f"  {'─'*60}")

    seq_last = all_results["sequential"][-1]["eval_all_gt"]["per_file_errors"]
    par_last = all_results["parallel"][-1]["eval_all_gt"]["per_file_errors"]

    for i, (se, pe) in enumerate(zip(seq_last, par_last)):
        ratio = pe / se if se > 0 else float("inf")
        print(f"  {'gt_' + str(i):<20} {se:>14.6f} {pe:>14.6f} {ratio:>10.4f}")

    print(f"  {'─'*60}")
    print(f"  {'MEAN':<20} {np.mean(seq_last):>14.6f} {np.mean(par_last):>14.6f} "
          f"{np.mean(par_last)/np.mean(seq_last) if np.mean(seq_last)>0 else float('inf'):>10.4f}")
    print()


if __name__ == "__main__":
    main()
