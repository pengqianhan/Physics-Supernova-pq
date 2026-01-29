#!/usr/bin/env python3
"""
Evaluate learned hybrid automata against ground truth data.

Usage: 
    python evaluate_result.py [--save-plots] [--output-dir DIR]
    python evaluate_result.py --gt0 [--save-plots] [--output-dir DIR]  # Evaluate with ground_truth_0.npz only
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
GT_0_FILENAME = "ground_truth_0.npz"  # For single ground_truth_0 evaluation


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


def evaluate_with_ground_truth_0(ha_spec_path: str, save_plot: bool = True) -> Dict[str, Any]:
    """
    Evaluate one HA spec against ground_truth_0.npz only.
    Results are saved in the same directory as the HA specification.
    
    Args:
        ha_spec_path: Path to best_ha_specification.json
        save_plot: Whether to save comparison plot
        
    Returns:
        Dictionary with evaluation results
    """
    with open(ha_spec_path, 'r') as f:
        ha_dict = json.load(f)
    
    gt_dir = get_ground_truth_dir(ha_spec_path)
    gt_file = os.path.join(gt_dir, GT_0_FILENAME)
    dataset_name = get_dataset_name(ha_spec_path)
    run_dir = os.path.dirname(ha_spec_path)
    
    if not os.path.exists(gt_file):
        return {'dataset': dataset_name, 'error': f'{GT_0_FILENAME} not found', 'status': 'failed'}
    
    # Set up plot path in the same run directory
    plot_path = os.path.join(run_dir, "ground_truth_0_overlay.png") if save_plot else None
    
    # Evaluate
    result = evaluate_single_file(ha_dict, gt_file, plot_path)
    
    # Add extra info
    num_iterations, max_iter = get_iteration_stats(ha_spec_path)
    result['dataset'] = dataset_name
    result['ha_spec_path'] = ha_spec_path
    result['num_iterations'] = num_iterations
    result['max_iter'] = max_iter
    
    # Save results to JSON and markdown in run directory
    json_path = os.path.join(run_dir, "ground_truth_0_eval.json")
    md_path = os.path.join(run_dir, "ground_truth_0_eval.md")
    
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Generate markdown content
    md_lines = [
        f"# Ground Truth 0 Evaluation: {dataset_name}",
        "",
        f"**HA Specification**: `{ha_spec_path}`",
        f"**Ground Truth File**: `{gt_file}`",
        f"**Iterations**: {num_iterations} (max_iter: {max_iter})",
        "",
        "## Metrics",
        "",
    ]
    
    if result['status'] == 'success':
        md_lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| tc (time correlation) | {result.get('tc', 'N/A')} |",
            f"| max_diff | {result.get('max_diff', 'N/A')} |",
            f"| mean_diff | {result.get('mean_diff', 'N/A')} |",
            f"| clustering_error | {result.get('clustering_error', 'N/A')} |",
            f"| status | {result['status']} |",
        ])
        if save_plot:
            md_lines.extend(["", f"![Overlay Plot](ground_truth_0_overlay.png)"])
    else:
        md_lines.extend([
            f"**Status**: {result['status']}",
            f"**Error**: {result.get('error', 'Unknown error')}",
        ])
    
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines) + "\n")
    
    return result


def evaluate_all_with_ground_truth_0(save_plots: bool = True, output_dir: str = "eval_outputs") -> List[Dict[str, Any]]:
    """
    Evaluate all HA specs against ground_truth_0.npz.
    Individual results are saved in each run directory.
    Summary is saved in output_dir.
    
    Args:
        save_plots: Whether to save comparison plots
        output_dir: Directory for summary files
        
    Returns:
        List of evaluation results
    """
    ha_specs = find_all_ha_specs()
    print(f"Found {len(ha_specs)} HA specifications to evaluate with {GT_0_FILENAME}")
    
    all_results = []
    for i, ha_spec_path in enumerate(ha_specs, 1):
        dataset_name = get_dataset_name(ha_spec_path)
        print(f"[{i}/{len(ha_specs)}] Evaluating {dataset_name} with {GT_0_FILENAME}...")
        
        result = evaluate_with_ground_truth_0(ha_spec_path, save_plots)
        all_results.append(result)
        
        if result['status'] == 'success':
            print(f"    max_diff: {result.get('max_diff', 'N/A'):.6f}, mean_diff: {result.get('mean_diff', 'N/A'):.6f}")
        else:
            print(f"    Error: {result.get('error', 'Unknown')}")
    
    # Save summary JSON
    os.makedirs(output_dir, exist_ok=True)
    summary_json = os.path.join(output_dir, 'ground_truth_0_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary JSON saved to: {summary_json}")
    
    # Build and save summary markdown
    valid = [r for r in all_results if r['status'] == 'success']
    summary_lines = [
        "# Ground Truth 0 Evaluation Summary",
        "",
        f"Total datasets: {len(all_results)}",
        f"Successful: {len(valid)}",
        f"Failed: {len(all_results) - len(valid)}",
        "",
        "## Results Table",
        "",
        "| Dataset | tc | max_diff | mean_diff | clustering_error | Iterations | max_iter | Status |",
        "|---------|----|---------:|----------:|-----------------:|-----------:|--------:|--------|",
    ]
    
    for r in all_results:
        if r['status'] == 'success':
            tc_val = f"{r.get('tc', 0):.6f}" if r.get('tc') is not None else "N/A"
            max_d = f"{r.get('max_diff', 0):.6f}" if r.get('max_diff') is not None else "N/A"
            mean_d = f"{r.get('mean_diff', 0):.6f}" if r.get('mean_diff') is not None else "N/A"
            clust_err = f"{r.get('clustering_error', 0):.6f}" if r.get('clustering_error') is not None else "N/A"
        else:
            tc_val = max_d = mean_d = clust_err = "N/A"
        
        summary_lines.append(
            f"| {r.get('dataset', 'Unknown'):<35} | {tc_val} | {max_d} | {mean_d} | {clust_err} | "
            f"{r.get('num_iterations', 0)} | {r.get('max_iter', 0)} | {r['status']} |"
        )
    
    # Add aggregate statistics if there are successful results
    if valid:
        summary_lines.extend([
            "",
            "## Aggregate Statistics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|-----:|----:|----:|----:|",
        ])
        for metric in ['tc', 'max_diff', 'mean_diff', 'clustering_error']:
            values = [r[metric] for r in valid if r.get(metric) is not None]
            if values:
                summary_lines.append(
                    f"| {metric} | {np.mean(values):.6f} | {np.std(values):.6f} | "
                    f"{np.min(values):.6f} | {np.max(values):.6f} |"
                )
    
    summary_md = os.path.join(output_dir, 'ground_truth_0_summary.md')
    with open(summary_md, 'w') as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Summary markdown saved to: {summary_md}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate HA specifications against ground truth')
    parser.add_argument('--save-plots', action='store_true', help='Save comparison plots')
    parser.add_argument('--output-dir', default='eval_outputs', help='Output directory')
    parser.add_argument('--gt0', action='store_true', help='Evaluate with ground_truth_0.npz only')
    args = parser.parse_args()

    # Handle ground_truth_0 evaluation mode
    if args.gt0:
        evaluate_all_with_ground_truth_0(save_plots=args.save_plots, output_dir=args.output_dir)
        return

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
