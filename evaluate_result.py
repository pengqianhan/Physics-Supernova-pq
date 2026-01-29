#!/usr/bin/env python3
"""
Evaluate learned hybrid automata against ground truth data.

Usage: python evaluate_result.py [--save-plots] [--output-dir DIR]
"""

import sys
import os
import json
import glob
import argparse
import numpy as np
from typing import Dict, List, Any, Optional

# Add Dainarx_code to path for HAEvaluator import
sys.path.insert(0, 'utils/Dainarx_code')
from HA_evaluation import HAEvaluator

EVAL_RESULTS_DIR = "evaluation_results_gemini3flash_260129"
DATA_DIR = "data_all"
GT_START_INDEX = 3  # Use ground_truth_3.npz onwards
GT_END_INDEX = 14   # Up to ground_truth_14.npz


def find_all_ha_specs() -> List[str]:
    """Find all best_ha_specification.json files."""
    pattern = f"{EVAL_RESULTS_DIR}/**/best_ha_specification.json"
    return sorted(glob.glob(pattern, recursive=True))


def get_ground_truth_dir(ha_spec_path: str) -> str:
    """
    Map HA spec path to ground truth directory.
    evaluation_results_gemini3flash_260129/ATVA/ball/runs/.../best_ha_specification.json
    -> data_all/ATVA/ball_g/
    """
    parts = ha_spec_path.split(os.sep)
    category = parts[1]  # e.g., "ATVA"
    dataset = parts[2]   # e.g., "ball"
    return os.path.join(DATA_DIR, category, f"{dataset}_g")


def get_dataset_name(ha_spec_path: str) -> str:
    """Extract dataset name like 'ATVA/ball' from path."""
    parts = ha_spec_path.split(os.sep)
    return f"{parts[1]}/{parts[2]}"


def get_ground_truth_files(gt_dir: str, start_idx: int = GT_START_INDEX) -> List[str]:
    """Get ground truth files starting from specified index."""
    files = []
    for i in range(start_idx, GT_END_INDEX + 1):
        path = os.path.join(gt_dir, f"ground_truth_{i}.npz")
        if os.path.exists(path):
            files.append(path)
    return sorted(files)


def get_iteration_stats(ha_spec_path: str) -> tuple:
    """
    Get iteration statistics from the same directory as the HA spec.

    Returns:
        tuple: (num_iterations, max_iter) where max_iter is the largest iter_N number
    """
    run_dir = os.path.dirname(ha_spec_path)
    iter_pattern = os.path.join(run_dir, "iter_*")
    iter_folders = glob.glob(iter_pattern)

    num_iterations = len(iter_folders)
    max_iter = 0

    for folder in iter_folders:
        folder_name = os.path.basename(folder)
        if folder_name.startswith("iter_"):
            try:
                iter_num = int(folder_name.split("_")[1])
                max_iter = max(max_iter, iter_num)
            except (ValueError, IndexError):
                pass

    return num_iterations, max_iter


def evaluate_single_file(ha_dict: Dict, gt_file: str, save_plot_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate one HA spec against one ground truth file."""
    dt = ha_dict['config'].get('dt', 0.001)
    total_time = ha_dict['config'].get('total_time', 10.0)

    try:
        evaluator = HAEvaluator(
            ha_dict=ha_dict,
            npz_file_path=gt_file,
            dt=dt,
            total_time=total_time
        )
        evaluator.load_ground_truth()
        evaluator.simulate()
        evaluator.compute_metrics()

        result = {
            'gt_file': os.path.basename(gt_file),
            'tc': evaluator.metrics['tc'],
            'max_diff': evaluator.metrics['max_diff'],
            'mean_diff': evaluator.metrics['mean_diff'],
            'clustering_error': evaluator.metrics['clustering_error'],
            'status': 'success'
        }

        if save_plot_path:
            evaluator.plot(plot_mode='overlay', save_path=save_plot_path)

        return result

    except Exception as e:
        return {
            'gt_file': os.path.basename(gt_file),
            'error': str(e),
            'status': 'failed'
        }


def evaluate_dataset(ha_spec_path: str, save_plots: bool = False, output_dir: str = "eval_outputs") -> Dict[str, Any]:
    """Evaluate one HA spec against all its ground truth files."""
    with open(ha_spec_path, 'r') as f:
        ha_dict = json.load(f)

    gt_dir = get_ground_truth_dir(ha_spec_path)
    gt_files = get_ground_truth_files(gt_dir)
    dataset_name = get_dataset_name(ha_spec_path)

    if not gt_files:
        return {'dataset': dataset_name, 'error': 'No ground truth files found', 'results': []}

    results = []
    for gt_file in gt_files:
        plot_path = None
        if save_plots:
            plot_dir = os.path.join(output_dir, dataset_name.replace('/', '_'))
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{os.path.basename(gt_file).replace('.npz', '.png')}")

        result = evaluate_single_file(ha_dict, gt_file, plot_path)
        results.append(result)

    # Compute aggregate metrics
    valid = [r for r in results if r['status'] == 'success']
    aggregate = {}
    if valid:
        for metric in ['tc', 'max_diff', 'mean_diff']:
            values = [r[metric] for r in valid if r[metric] is not None]
            if values:
                aggregate[f'{metric}_mean'] = float(np.mean(values))
                aggregate[f'{metric}_std'] = float(np.std(values))
        aggregate['success_rate'] = len(valid) / len(results)

    num_iterations, max_iter = get_iteration_stats(ha_spec_path)

    return {
        'dataset': dataset_name,
        'ha_spec_path': ha_spec_path,
        'num_files': len(gt_files),
        'num_success': len(valid),
        'num_iterations': num_iterations,
        'max_iter': max_iter,
        'aggregate': aggregate,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate HA specifications against ground truth')
    parser.add_argument('--save-plots', action='store_true', help='Save comparison plots')
    parser.add_argument('--output-dir', default='eval_outputs', help='Output directory')
    args = parser.parse_args()

    ha_specs = find_all_ha_specs()
    print(f"Found {len(ha_specs)} HA specifications to evaluate")

    all_results = []
    for i, ha_spec_path in enumerate(ha_specs, 1):
        dataset_name = get_dataset_name(ha_spec_path)
        print(f"[{i}/{len(ha_specs)}] Evaluating {dataset_name}...")

        result = evaluate_dataset(ha_spec_path, args.save_plots, args.output_dir)
        all_results.append(result)

        # Print summary for this dataset
        if 'aggregate' in result and result['aggregate']:
            agg = result['aggregate']
            print(f"    max_diff: {agg.get('max_diff_mean', 'N/A'):.6f} +/- {agg.get('max_diff_std', 0):.6f}")
            print(f"    mean_diff: {agg.get('mean_diff_mean', 'N/A'):.6f} +/- {agg.get('mean_diff_std', 0):.6f}")

    # Save detailed results
    output_json = os.path.join(args.output_dir, 'evaluation_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_json}")

    # Build and print final summary table
    summary_lines = []
    summary_lines.append("=" * 120)
    summary_lines.append("EVALUATION SUMMARY")
    summary_lines.append("=" * 120)
    summary_lines.append(f"{'Dataset':<40} {'tc':<15} {'max_diff':<15} {'mean_diff':<15} {'Success':<10} {'Iterations':<12} {'max_iter'}")
    summary_lines.append("-" * 120)
    for r in all_results:
        agg = r.get('aggregate', {})
        tc_val = f"{agg.get('tc_mean', 0):.6f}" if agg else "N/A"
        max_d = f"{agg.get('max_diff_mean', 0):.6f}" if agg else "N/A"
        mean_d = f"{agg.get('mean_diff_mean', 0):.6f}" if agg else "N/A"
        success = f"{r.get('num_success', 0)}/{r.get('num_files', 0)}"
        iterations = str(r.get('num_iterations', 0))
        max_iter = str(r.get('max_iter', 0))
        summary_lines.append(f"{r['dataset']:<40} {tc_val:<15} {max_d:<15} {mean_d:<15} {success:<10} {iterations:<12} {max_iter}")
    summary_lines.append("=" * 120)

    # Print to console
    print("\n" + "\n".join(summary_lines))

    # Save summary table to file
    summary_txt = os.path.join(args.output_dir, 'evaluation_summary.md')
    with open(summary_txt, 'w') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary table saved to: {summary_txt}")


if __name__ == "__main__":
    main()
