#!/usr/bin/env python3
"""
Analyze ground_truth_0_eval.json files and find datasets with mean_diff < 0.001
"""

import json
from pathlib import Path


def analyze_evaluation_results(base_dir: str = "evaluation_results_gemini3flash_260129",
                                threshold: float = 0.001,
                                output_file: str = "ground_truth_0_analysis.md"):
    """
    Scan all ground_truth_0_eval.json files and record datasets with mean_diff < threshold.

    Args:
        base_dir: Directory containing evaluation results
        threshold: mean_diff threshold (default 0.001)
        output_file: Output markdown file path
    """
    base_path = Path(base_dir)

    # Find all ground_truth_0_eval.json files
    eval_files = list(base_path.glob("**/ground_truth_0_eval.json"))

    passing_datasets = []
    failing_datasets = []

    for eval_file in sorted(eval_files):
        try:
            with open(eval_file, 'r') as f:
                data = json.load(f)

            mean_diff = data.get("mean_diff")
            dataset = data.get("dataset", str(eval_file.parent))
            status = data.get("status", "unknown")

            if mean_diff is not None and mean_diff < threshold:
                passing_datasets.append({
                    "dataset": dataset,
                    "mean_diff": mean_diff,
                    "max_diff": data.get("max_diff"),
                    "status": status,
                    "file_path": str(eval_file)
                })
            else:
                failing_datasets.append({
                    "dataset": dataset,
                    "mean_diff": mean_diff,
                    "max_diff": data.get("max_diff"),
                    "status": status,
                    "file_path": str(eval_file)
                })
        except Exception as e:
            print(f"Error processing {eval_file}: {e}")

    # Generate markdown report
    md_lines = [
        "# Ground Truth 0 Evaluation Analysis",
        "",
        f"**Threshold**: mean_diff < {threshold}",
        f"**Total datasets**: {len(eval_files)}",
        f"**Passing**: {len(passing_datasets)}",
        f"**Failing**: {len(failing_datasets)}",
        "",
        "## Passing Datasets (mean_diff < 0.001)",
        "",
        "| Dataset | mean_diff | max_diff | status |",
        "|---------|-----------|----------|--------|",
    ]

    for item in passing_datasets:
        md_lines.append(
            f"| {item['dataset']} | {item['mean_diff']:.6f} | {item['max_diff']:.6f} | {item['status']} |"
        )

    md_lines.extend([
        "",
        "## Failing Datasets (mean_diff >= 0.001)",
        "",
        "| Dataset | mean_diff | max_diff | status |",
        "|---------|-----------|----------|--------|",
    ])

    for item in failing_datasets:
        mean_diff_str = f"{item['mean_diff']:.6f}" if item['mean_diff'] is not None else "N/A"
        max_diff_str = f"{item['max_diff']:.6f}" if item['max_diff'] is not None else "N/A"
        md_lines.append(
            f"| {item['dataset']} | {mean_diff_str} | {max_diff_str} | {item['status']} |"
        )

    # Write markdown file
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Passing datasets: {len(passing_datasets)}/{len(eval_files)}")

    return passing_datasets, failing_datasets


if __name__ == "__main__":
    analyze_evaluation_results()
