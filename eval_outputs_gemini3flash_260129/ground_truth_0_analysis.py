#!/usr/bin/env python3
"""
Analyze ground_truth_0_summary.json and find datasets with mean_diff < 0.001
"""

import json
from pathlib import Path


def analyze_from_summary(summary_file: str = None,
                         threshold: float = 0.001,
                         output_file: str = None):
    # Default paths relative to script location
    script_dir = Path(__file__).parent
    if summary_file is None:
        summary_file = script_dir / "ground_truth_0_summary.json"
    if output_file is None:
        output_file = script_dir / "ground_truth_0_analysis.md"
    """
    Read ground_truth_0_summary.json and record datasets with mean_diff < threshold.

    Args:
        summary_file: Path to the summary JSON file
        threshold: mean_diff threshold (default 0.001)
        output_file: Output markdown file path
    """
    with open(summary_file, 'r') as f:
        data = json.load(f)

    passing_datasets = []
    failing_datasets = []

    for item in data:
        mean_diff = item.get("mean_diff")
        dataset = item.get("dataset", "unknown")
        status = item.get("status", "unknown")

        entry = {
            "dataset": dataset,
            "mean_diff": mean_diff,
            "max_diff": item.get("max_diff"),
            "tc": item.get("tc"),
            "clustering_error": item.get("clustering_error"),
            "num_iterations": item.get("num_iterations"),
            "max_iter": item.get("max_iter"),
            "status": status,
            "ha_spec_path": item.get("ha_spec_path")
        }

        if mean_diff is not None and mean_diff < threshold:
            passing_datasets.append(entry)
        else:
            failing_datasets.append(entry)

    # Sort passing by mean_diff
    passing_datasets.sort(key=lambda x: x['mean_diff'])
    failing_datasets.sort(key=lambda x: x['mean_diff'] if x['mean_diff'] is not None else float('inf'))

    # Generate markdown report
    md_lines = [
        "# Ground Truth 0 Evaluation Analysis",
        "",
        f"**Threshold**: mean_diff < {threshold}",
        f"**Total datasets**: {len(data)}",
        f"**Passing**: {len(passing_datasets)}",
        f"**Failing**: {len(failing_datasets)}",
        f"**Success rate**: {len(passing_datasets)/len(data)*100:.1f}%",
        "",
        "## Summary Statistics",
        "",
        f"- Datasets with perfect match (mean_diff = 0): {sum(1 for d in data if d.get('mean_diff') == 0)}",
        f"- Datasets with clustering errors: {sum(1 for d in data if d.get('clustering_error', 0) > 0)}",
        "",
        "---",
        "",
        "## Passing Datasets (mean_diff < 0.001)",
        "",
        "| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |",
        "|---------|-----------|----------|-----|------------------|------------|",
    ]

    for item in passing_datasets:
        md_lines.append(
            f"| {item['dataset']} | {item['mean_diff']:.6e} | {item['max_diff']:.6e} | {item['tc']} | {item['clustering_error']} | {item['num_iterations']}/{item['max_iter']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Failing Datasets (mean_diff >= 0.001)",
        "",
        "| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |",
        "|---------|-----------|----------|-----|------------------|------------|",
    ])

    for item in failing_datasets:
        mean_diff_str = f"{item['mean_diff']:.6e}" if item['mean_diff'] is not None else "N/A"
        max_diff_str = f"{item['max_diff']:.6e}" if item['max_diff'] is not None else "N/A"
        md_lines.append(
            f"| {item['dataset']} | {mean_diff_str} | {max_diff_str} | {item['tc']} | {item['clustering_error']} | {item['num_iterations']}/{item['max_iter']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Passing Datasets Details",
        "",
    ])

    for item in passing_datasets:
        md_lines.extend([
            f"### {item['dataset']}",
            "",
            f"- **mean_diff**: {item['mean_diff']:.6e}",
            f"- **max_diff**: {item['max_diff']:.6e}",
            f"- **tc**: {item['tc']}",
            f"- **clustering_error**: {item['clustering_error']}",
            f"- **iterations**: {item['num_iterations']}/{item['max_iter']}",
            f"- **HA spec**: `{item['ha_spec_path']}`",
            "",
        ])

    # Write markdown file
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Passing datasets: {len(passing_datasets)}/{len(data)} ({len(passing_datasets)/len(data)*100:.1f}%)")

    return passing_datasets, failing_datasets


if __name__ == "__main__":
    analyze_from_summary()
