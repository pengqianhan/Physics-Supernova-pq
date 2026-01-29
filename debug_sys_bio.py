#!/usr/bin/env python3
"""
Debug script for non_linear/sys_bio NaN issues.

This script analyzes why sys_bio evaluation produces NaN metrics and verifies the fix.

Usage: python debug_sys_bio.py

Results are self-contained - no HAEvaluator execution needed (uses data from evaluation_summary.json).
"""

import numpy as np


def main():
    # Data from evaluation_summary.json for non_linear/sys_bio
    results = [
        {"gt_file": "ground_truth_10.npz", "tc": 1.507, "max_diff": 1.270109197379771, "mean_diff": 0.3869838372827008},
        {"gt_file": "ground_truth_11.npz", "tc": 1.789, "max_diff": 0.9927549106146336, "mean_diff": 0.3228759682135301},
        {"gt_file": "ground_truth_12.npz", "tc": 0.0, "max_diff": 1.6094524758955864, "mean_diff": 0.5356296709767276},
        {"gt_file": "ground_truth_13.npz", "tc": 1.489, "max_diff": 6.004258688381948, "mean_diff": 0.4627683032567954},
        {"gt_file": "ground_truth_14.npz", "tc": 2.143, "max_diff": 0.9981715513045839, "mean_diff": 0.3163286132646405},
        {"gt_file": "ground_truth_3.npz", "tc": 1.451, "max_diff": 1.3422373888513692, "mean_diff": 0.35242167235063904},
        {"gt_file": "ground_truth_4.npz", "tc": 1.345, "max_diff": 0.9872986102176426, "mean_diff": 0.3396892619034521},
        {"gt_file": "ground_truth_5.npz", "tc": 3.284, "max_diff": float('nan'), "mean_diff": float('nan')},  # NaN!
        {"gt_file": "ground_truth_6.npz", "tc": 1.562, "max_diff": 1.0185916912325286, "mean_diff": 0.38286614478753755},
        {"gt_file": "ground_truth_7.npz", "tc": 1.379, "max_diff": 0.9884487290509235, "mean_diff": 0.3378367505601836},
        {"gt_file": "ground_truth_8.npz", "tc": 1.641, "max_diff": 0.985341331404206, "mean_diff": 0.339668888627636},
        {"gt_file": "ground_truth_9.npz", "tc": 1.706, "max_diff": 0.9940284008281177, "mean_diff": 0.3332272505766045},
    ]

    print("=" * 70)
    print("Debug Script: non_linear/sys_bio NaN Issue Analysis")
    print("=" * 70)

    # Phase 1: Show individual results
    print("\n" + "=" * 70)
    print("PHASE 1: Individual File Results")
    print("=" * 70)
    print(f"{'File':<25} {'max_diff':>12} {'mean_diff':>12} {'Status':>15}")
    print("-" * 70)

    nan_files = []
    for r in results:
        has_nan = np.isnan(r['max_diff']) or np.isnan(r['mean_diff'])
        status = "*** NaN ***" if has_nan else "OK"
        if has_nan:
            nan_files.append(r['gt_file'])
        print(f"{r['gt_file']:<25} {r['max_diff']:>12.6f} {r['mean_diff']:>12.6f} {status:>15}")

    # Phase 2: Root cause analysis
    print("\n" + "=" * 70)
    print("PHASE 2: Root Cause Analysis")
    print("=" * 70)
    print(f"""
Files with NaN metrics: {nan_files}

Root cause for ground_truth_5.npz:
- The ground truth data itself contains NaN values
- State variables 5, 7, 8 diverge to ~10^138 at t=6.878s
- Then overflow to NaN
- This is numerical instability in the original ODE simulation

Impact:
- HAEvaluator computes max_diff/mean_diff which become NaN
  when comparing against ground truth containing NaN
""")

    # Phase 3: Aggregation comparison
    print("=" * 70)
    print("PHASE 3: Aggregation Methods Comparison")
    print("=" * 70)

    # Method 1: Original (buggy)
    print("\n--- Method 1: Original evaluate_result.py (BUGGY) ---")
    print("Code: values = [r[metric] for r in valid if r[metric] is not None]")
    print("      aggregate[metric_mean] = np.mean(values)")
    print()
    for metric in ['max_diff', 'mean_diff']:
        values = [r[metric] for r in results if r[metric] is not None]
        mean_val = float(np.mean(values))
        print(f"  {metric}_mean = {mean_val}")
    print("\n  Problem: 'r[metric] is not None' does NOT filter NaN")
    print("           np.mean([..., NaN, ...]) = NaN")

    # Method 2: Fixed with explicit filter
    print("\n--- Method 2: Fixed (filter NaN explicitly) ---")
    print("Code: values = [r[metric] for r in valid if r[metric] is not None and not np.isnan(r[metric])]")
    print()
    for metric in ['max_diff', 'mean_diff']:
        values = [r[metric] for r in results if r[metric] is not None and not np.isnan(r[metric])]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        print(f"  {metric}_mean = {mean_val:.6f} +/- {std_val:.6f} (n={len(values)}/12)")

    # Method 3: Using np.nanmean
    print("\n--- Method 3: Using np.nanmean ---")
    print("Code: aggregate[metric_mean] = np.nanmean(values)")
    print()
    for metric in ['max_diff', 'mean_diff']:
        values = [r[metric] for r in results if r[metric] is not None]
        mean_val = float(np.nanmean(values))
        std_val = float(np.nanstd(values))
        print(f"  {metric}_mean = {mean_val:.6f} +/- {std_val:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("""
Problem:
--------
1. ground_truth_5.npz has NaN in state data (ODE solver divergence)
2. evaluate_result.py's NaN check (is not None) doesn't catch NaN
3. np.mean() propagates NaN to aggregate results

Fixes (choose one):
-------------------
Option A (Recommended): Filter NaN values explicitly
  Change line 152 in evaluate_result.py from:
    values = [r[metric] for r in valid if r[metric] is not None]
  To:
    values = [r[metric] for r in valid if r[metric] is not None and not np.isnan(r[metric])]

Option B: Use np.nanmean instead of np.mean
  Change line 154 in evaluate_result.py from:
    aggregate[f'{metric}_mean'] = float(np.mean(values))
  To:
    aggregate[f'{metric}_mean'] = float(np.nanmean(values))

Option C: Mark ground truth files with NaN as invalid during evaluation
  Add a pre-check in evaluate_single_file() to detect NaN in ground truth

Corrected values for non_linear/sys_bio:
-----------------------------------------
  max_diff_mean  = 1.562790 (was NaN)
  mean_diff_mean = 0.373663 (was NaN)
""")


if __name__ == "__main__":
    main()
