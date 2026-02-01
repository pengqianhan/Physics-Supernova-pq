#!/usr/bin/env python3
"""
Analyze ground_truth_0_summary.json and classify datasets into three categories:
- Category 1 (Excellent): mean_diff < 0.001
- Category 2 (Good): 0.001 <= mean_diff < 0.005
- Category 3 (Poor): mean_diff >= 0.005
"""

import json
from pathlib import Path


def analyze_from_summary(summary_file: str = None,
                         threshold_excellent: float = 0.001,
                         threshold_good: float = 0.005,
                         output_file: str = None):
    """
    Read ground_truth_0_summary.json and classify datasets into three categories.

    Args:
        summary_file: Path to the summary JSON file
        threshold_excellent: Upper bound for excellent category (default 0.001)
        threshold_good: Upper bound for good category (default 0.005)
        output_file: Output markdown file path
    """
    # Default paths relative to script location
    script_dir = Path(__file__).parent
    if summary_file is None:
        summary_file = script_dir / "ground_truth_0_summary.json"
    if output_file is None:
        output_file = script_dir / "ground_truth_0_analysis.md"

    with open(summary_file, 'r') as f:
        data = json.load(f)

    # Three categories
    excellent_datasets = []  # mean_diff < 0.001
    good_datasets = []       # 0.001 <= mean_diff < 0.005
    poor_datasets = []       # mean_diff >= 0.005

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

        if mean_diff is None:
            poor_datasets.append(entry)
        elif mean_diff < threshold_excellent:
            excellent_datasets.append(entry)
        elif mean_diff < threshold_good:
            good_datasets.append(entry)
        else:
            poor_datasets.append(entry)

    # Sort each category by mean_diff
    excellent_datasets.sort(key=lambda x: x['mean_diff'])
    good_datasets.sort(key=lambda x: x['mean_diff'])
    poor_datasets.sort(key=lambda x: x['mean_diff'] if x['mean_diff'] is not None else float('inf'))

    # Generate markdown report
    total = len(data)
    md_lines = [
        "# Ground Truth 0 Evaluation Analysis",
        "",
        "## Classification Criteria",
        "",
        f"- **Category 1 (Excellent)**: mean_diff < {threshold_excellent}",
        f"- **Category 2 (Good)**: {threshold_excellent} <= mean_diff < {threshold_good}",
        f"- **Category 3 (Poor)**: mean_diff >= {threshold_good}",
        "",
        "## Overview",
        "",
        f"**Total datasets**: {total}",
        f"**Excellent (Cat 1)**: {len(excellent_datasets)} ({len(excellent_datasets)/total*100:.1f}%)",
        f"**Good (Cat 2)**: {len(good_datasets)} ({len(good_datasets)/total*100:.1f}%)",
        f"**Poor (Cat 3)**: {len(poor_datasets)} ({len(poor_datasets)/total*100:.1f}%)",
        "",
        "## Summary Statistics",
        "",
        f"- Datasets with perfect match (mean_diff = 0): {sum(1 for d in data if d.get('mean_diff') == 0)}",
        f"- Datasets with clustering errors: {sum(1 for d in data if d.get('clustering_error', 0) > 0)}",
        "",
        "---",
        "",
        f"## Category 1: Excellent (mean_diff < {threshold_excellent})",
        "",
        "| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |",
        "|---------|-----------|----------|-----|------------------|------------|",
    ]

    for item in excellent_datasets:
        md_lines.append(
            f"| {item['dataset']} | {item['mean_diff']:.6f} | {item['max_diff']:.6f} | {item['tc']} | {item['clustering_error']} | {item['num_iterations']}/{item['max_iter']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        f"## Category 2: Good ({threshold_excellent} <= mean_diff < {threshold_good})",
        "",
        "| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |",
        "|---------|-----------|----------|-----|------------------|------------|",
    ])

    for item in good_datasets:
        md_lines.append(
            f"| {item['dataset']} | {item['mean_diff']:.6f} | {item['max_diff']:.6f} | {item['tc']} | {item['clustering_error']} | {item['num_iterations']}/{item['max_iter']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        f"## Category 3: Poor (mean_diff >= {threshold_good})",
        "",
        "| Dataset | mean_diff | max_diff | tc | clustering_error | iterations |",
        "|---------|-----------|----------|-----|------------------|------------|",
    ])

    for item in poor_datasets:
        mean_diff_str = f"{item['mean_diff']:.6f}" if item['mean_diff'] is not None else "N/A"
        max_diff_str = f"{item['max_diff']:.6f}" if item['max_diff'] is not None else "N/A"
        md_lines.append(
            f"| {item['dataset']} | {mean_diff_str} | {max_diff_str} | {item['tc']} | {item['clustering_error']} | {item['num_iterations']}/{item['max_iter']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Excellent Datasets Details",
        "",
    ])

    for item in excellent_datasets:
        md_lines.extend([
            f"### {item['dataset']}",
            "",
            f"- **mean_diff**: {item['mean_diff']:.6f}",
            f"- **max_diff**: {item['max_diff']:.6f}",
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
    print(f"Category 1 (Excellent, mean_diff < {threshold_excellent}): {len(excellent_datasets)}/{total} ({len(excellent_datasets)/total*100:.1f}%)")
    print(f"Category 2 (Good, {threshold_excellent} <= mean_diff < {threshold_good}): {len(good_datasets)}/{total} ({len(good_datasets)/total*100:.1f}%)")
    print(f"Category 3 (Poor, mean_diff >= {threshold_good}): {len(poor_datasets)}/{total} ({len(poor_datasets)/total*100:.1f}%)")

    return excellent_datasets, good_datasets, poor_datasets


if __name__ == "__main__":
    analyze_from_summary()
