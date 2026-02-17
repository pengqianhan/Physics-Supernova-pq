"""
Test: Run optimize_ha_params.py in parallel mode only.

Uses default parallel settings:
  num_workers = max(1, min(16, cpu_count // 4))
  budget = base_budget * sqrt(num_workers)

Metrics reported:
  - Wall-clock time (seconds)
  - Final optimization error (mean_diff)
  - Per-ground-truth evaluation errors

Usage:
    python test_parallel_only.py
    python test_parallel_only.py --trials 3        # repeat N times
    python test_parallel_only.py --train_num 2     # use 2 GT files
"""

import argparse
import os
import re
import sys
import time
import warnings
from typing import Any, Dict

warnings.filterwarnings("ignore")

import numpy as np

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
        except Exception:
            errors.append(float("inf"))

    return {
        "per_file_errors": errors,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "num_files": len(errors),
    }


# ── Single parallel run ──────────────────────────────────────────────
def run_once(train_num: int = 3, label: str = "") -> Dict[str, Any]:
    """Run optimization with default parallel workers and return results + timing."""
    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(16, cpu_count // 4))

    print(f"\n{'='*70}")
    print(f"  RUN: {label}")
    print(f"  num_workers = {default_workers} (auto)")
    print(f"{'='*70}")

    t0 = time.perf_counter()

    result = optimize_ha_parameters_generic(
        ha_spec=DUFFING_HA_SPEC,
        input_data_path=INPUT_DATA_PATH,
        model_id="gemini/gemini-3-flash-preview",
        verbose=True,
        train_num=train_num,
        num_workers=None,  # use default parallel
    )

    elapsed = time.perf_counter() - t0

    print(f"\n[Eval] Evaluating optimized spec on all ground truth files...")
    eval_results = evaluate_on_all_gt(result["ha_spec"], INPUT_DATA_PATH)

    return {
        "label": label,
        "num_workers": default_workers,
        "train_error": result["error"],
        "wall_time_s": elapsed,
        "parameters": result["parameters"],
        "eval_all_gt": eval_results,
    }


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run parallel HA optimization")
    parser.add_argument("--trials", type=int, default=3, help="Number of repeated trials (default: 3)")
    parser.add_argument("--train_num", type=int, default=3, help="Number of GT files for training (default: 3)")
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 1
    default_workers = max(1, min(16, cpu_count // 4))

    print("#" * 70)
    print("#  TEST: Parallel HA Optimization")
    print(f"#  CPU count: {cpu_count}")
    print(f"#  num_workers: {default_workers}")
    print(f"#  Trials: {args.trials}")
    print(f"#  Train files: {args.train_num}")
    print("#" * 70)

    results = []

    for trial in range(args.trials):
        print(f"\n\n{'#'*70}")
        print(f"#  TRIAL {trial + 1} / {args.trials}")
        print(f"{'#'*70}")

        r = run_once(
            train_num=args.train_num,
            label=f"Parallel (num_workers={default_workers}) [trial {trial+1}]",
        )
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n")
    print("=" * 80)
    print(f"  PARALLEL OPTIMIZATION SUMMARY  (num_workers={default_workers})")
    print("=" * 80)

    times = [r["wall_time_s"] for r in results]
    train_errors = [r["train_error"] for r in results]
    eval_means = [r["eval_all_gt"]["mean_error"] for r in results]

    if args.trials == 1:
        r = results[0]
        print(f"    num_workers:        {r['num_workers']}")
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

        # Best trial
        best_idx = int(np.argmin(eval_means))
        best = results[best_idx]
        print(f"\n    Best trial: #{best_idx + 1}")
        print(f"      Train error:      {best['train_error']:.6f}")
        print(f"      Eval mean error:  {best['eval_all_gt']['mean_error']:.6f}")
        print(f"      Wall-clock time:  {best['wall_time_s']:.2f} s")
        print(f"      Optimized params:")
        for name, val in best["parameters"].items():
            print(f"        {name}: {val:.6f}")

    # ── Per-GT-file detail table (last trial) ─────────────────────────
    print(f"\n  PER-GROUND-TRUTH FILE ERRORS (last trial):")
    print(f"  {'GT File':<20} {'Error':>14}")
    print(f"  {'─'*36}")

    last_errors = results[-1]["eval_all_gt"]["per_file_errors"]
    for i, err in enumerate(last_errors):
        print(f"  {'gt_' + str(i):<20} {err:>14.6f}")

    print(f"  {'─'*36}")
    print(f"  {'MEAN':<20} {np.mean(last_errors):>14.6f}")
    print()


if __name__ == "__main__":
    main()
