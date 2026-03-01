#!/usr/bin/env python3
"""
Evaluate all HA JSON specifications under utils/Dainarx_code/automata/
against ground_truth_0, ground_truth_1, ground_truth_2 NPZ files.

Reports averaged metrics (mean_diff, max_diff, tc) per automaton.
All simulations start from mode 1.
"""

import json
import os
import sys
import numpy as np
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.Dainarx_code.HA_evaluation import HAEvaluator

AUTOMATA_DIR = os.path.join(PROJECT_ROOT, "utils", "Dainarx_code", "automata")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_all")
GT_INDICES = [0, 1, 2]


def find_all_automata():
    """Walk automata dir and return list of (benchmark, system_name, json_path)."""
    results = []
    for benchmark in sorted(os.listdir(AUTOMATA_DIR)):
        bench_dir = os.path.join(AUTOMATA_DIR, benchmark)
        if not os.path.isdir(bench_dir):
            continue
        # skip non-benchmark dirs like json_readme.md etc
        for fname in sorted(os.listdir(bench_dir)):
            if not fname.endswith(".json"):
                continue
            system_name = fname[:-5]  # strip .json
            json_path = os.path.join(bench_dir, fname)
            results.append((benchmark, system_name, json_path))
    return results


def evaluate_single(ha_dict, npz_path):
    """
    Run HAEvaluator for a single JSON + single NPZ file.
    Returns metrics dict or None on failure.
    """
    dt = ha_dict["config"]["dt"]
    total_time = ha_dict["config"]["total_time"]

    evaluator = HAEvaluator(
        ha_dict=ha_dict,
        npz_file_path=npz_path,
        dt=dt,
        total_time=total_time,
    )
    evaluator.load_ground_truth()
    evaluator.simulate()
    evaluator.compute_metrics()
    return evaluator.metrics


def average_metrics(metrics_list):
    """Average numeric metric keys across a list of metric dicts."""
    keys = ["mean_diff", "max_diff", "tc", "train_tc", "clustering_error"]
    avg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if vals:
            avg[k] = sum(vals) / len(vals)
        else:
            avg[k] = None
    return avg


def main():
    automata = find_all_automata()
    print(f"Found {len(automata)} automata to evaluate.\n")
    print(f"{'Benchmark':<16} {'System':<30} {'mean_diff':>12} {'max_diff':>12} {'tc':>12} {'train_tc':>12} {'clust_err':>12}  Status")
    print("-" * 130)

    all_results = []

    for benchmark, system_name, json_path in automata:
        gt_dir = os.path.join(DATA_DIR, benchmark, f"{system_name}_g")

        # Check ground truth dir exists
        if not os.path.isdir(gt_dir):
            print(f"{benchmark:<16} {system_name:<30} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'—':>12}  SKIP: no gt dir")
            continue

        # Load JSON
        with open(json_path, "r") as f:
            ha_dict = json.load(f)

        # Ensure automaton has 'input' key (some JSONs omit it)
        if "input" not in ha_dict["automaton"]:
            ha_dict["automaton"]["input"] = ""

        # Evaluate against each ground truth file
        metrics_list = []
        missing = []
        errors = []

        for idx in GT_INDICES:
            npz_path = os.path.join(gt_dir, f"ground_truth_{idx}.npz")
            if not os.path.isfile(npz_path):
                missing.append(idx)
                continue
            try:
                m = evaluate_single(ha_dict, npz_path)
                metrics_list.append(m)
            except Exception as e:
                errors.append((idx, str(e)))

        if not metrics_list:
            reason = ""
            if missing:
                reason += f"missing gt {missing} "
            if errors:
                reason += f"errors: {errors[0][1][:60]}"
            print(f"{benchmark:<16} {system_name:<30} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'—':>12}  FAIL: {reason}")
            continue

        avg = average_metrics(metrics_list)
        status = "OK"
        if missing:
            status += f" (missing gt {missing})"
        if errors:
            status += f" ({len(errors)} errors)"

        def fmt(v):
            return f"{v:.6f}" if v is not None else "N/A"

        print(f"{benchmark:<16} {system_name:<30} {fmt(avg['mean_diff']):>12} {fmt(avg['max_diff']):>12} {fmt(avg['tc']):>12} {fmt(avg['train_tc']):>12} {fmt(avg.get('clustering_error')):>12}  {status}")

        all_results.append({
            "benchmark": benchmark,
            "system": system_name,
            "n_evaluated": len(metrics_list),
            "avg_metrics": avg,
        })

    # Summary
    print("\n" + "=" * 130)
    print(f"Total: {len(all_results)} automata evaluated successfully out of {len(automata)} found.")

    # Overall averages (only over systems that succeeded)
    if all_results:
        overall_mean = []
        overall_max = []
        for r in all_results:
            if r["avg_metrics"]["mean_diff"] is not None:
                overall_mean.append(r["avg_metrics"]["mean_diff"])
            if r["avg_metrics"]["max_diff"] is not None:
                overall_max.append(r["avg_metrics"]["max_diff"])
        if overall_mean:
            print(f"Overall average mean_diff: {sum(overall_mean)/len(overall_mean):.6f}")
        if overall_max:
            print(f"Overall average max_diff:  {sum(overall_max)/len(overall_max):.6f}")


if __name__ == "__main__":
    main()
