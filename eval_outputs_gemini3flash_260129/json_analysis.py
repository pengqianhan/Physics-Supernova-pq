import json

with open('eval_outputs_gemini3flash_260129/evaluation_summary.json', 'r') as f:
    data = json.load(f)

# Categorize datasets by mean_diff_mean
# Category 1: mean_diff_mean <= 0.001 (Excellent)
# Category 2: 0.001 < mean_diff_mean <= 0.1 (Good)
# Category 3: mean_diff_mean > 0.1 (Poor)

category_1 = []  # <= 0.001
category_2 = []  # (0.001, 0.1]
category_3 = []  # > 0.1

for entry in data:
    dataset = entry['dataset']
    mean_diff_mean = entry['aggregate']['mean_diff_mean']
    
    record = {
        'dataset': dataset,
        'mean_diff_mean': mean_diff_mean,
        'max_diff_mean': entry['aggregate']['max_diff_mean'],
        'tc_mean': entry['aggregate']['tc_mean'],
        'success_rate': entry['aggregate']['success_rate'],
        'max_iter': entry['max_iter']
    }
    
    if mean_diff_mean <= 0.001:
        category_1.append(record)
    elif mean_diff_mean <= 0.1:
        category_2.append(record)
    else:
        category_3.append(record)

# Sort each category by mean_diff_mean
category_1.sort(key=lambda x: x['mean_diff_mean'])
category_2.sort(key=lambda x: x['mean_diff_mean'])
category_3.sort(key=lambda x: x['mean_diff_mean'])

# Generate LaTeX table
def format_scientific(val):
    """Format small values in scientific notation, others normally"""
    if val == 0:
        return "0"
    elif val < 0.0001:
        return f"{val:.2e}"
    elif val < 0.01:
        return f"{val:.4f}"
    else:
        return f"{val:.3f}"

def generate_latex_table(category_1, category_2, category_3):
    """Generate a LaTeX table for dataset classification"""
    latex_output = []
    latex_output.append(r"\begin{table}[htbp]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Dataset Classification by Mean Difference (Gemini 3 Flash)}")
    latex_output.append(r"\label{tab:dataset_classification}")
    latex_output.append(r"\begin{tabular}{l|c|c|c|c}")
    latex_output.append(r"\hline")
    latex_output.append(r"\textbf{Dataset} & \textbf{Mean Diff} & \textbf{Max Diff} & \textbf{TC Mean} & \textbf{Max Iter} \\")
    latex_output.append(r"\hline")

    # Category 1: Excellent (mean_diff_mean <= 0.001)
    latex_output.append(r"\multicolumn{5}{c}{\textbf{Category 1: Excellent ($\bar{\epsilon} \leq 0.001$)}} \\")
    latex_output.append(r"\hline")
    for r in category_1:
        name = r['dataset'].replace('_', r'\_')
        latex_output.append(f"{name} & {format_scientific(r['mean_diff_mean'])} & {format_scientific(r['max_diff_mean'])} & {r['tc_mean']:.3f} & {r['max_iter']} \\\\")
    latex_output.append(r"\hline")

    # Category 2: Good (0.001 < mean_diff_mean <= 0.1)
    latex_output.append(r"\multicolumn{5}{c}{\textbf{Category 2: Good ($0.001 < \bar{\epsilon} \leq 0.1$)}} \\")
    latex_output.append(r"\hline")
    for r in category_2:
        name = r['dataset'].replace('_', r'\_')
        latex_output.append(f"{name} & {format_scientific(r['mean_diff_mean'])} & {format_scientific(r['max_diff_mean'])} & {r['tc_mean']:.3f} & {r['max_iter']} \\\\")
    latex_output.append(r"\hline")

    # Category 3: Poor (mean_diff_mean > 0.1)
    latex_output.append(r"\multicolumn{5}{c}{\textbf{Category 3: Poor ($\bar{\epsilon} > 0.1$)}} \\")
    latex_output.append(r"\hline")
    for r in category_3:
        name = r['dataset'].replace('_', r'\_')
        latex_output.append(f"{name} & {format_scientific(r['mean_diff_mean'])} & {format_scientific(r['max_diff_mean'])} & {r['tc_mean']:.3f} & {r['max_iter']} \\\\")
    latex_output.append(r"\hline")

    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    
    return "\n".join(latex_output)

# Generate LaTeX table
latex_output = generate_latex_table(category_1, category_2, category_3)

# Print summary
print("=" * 60)
print("Dataset Classification Summary")
print("=" * 60)
print(f"\nCategory 1 (mean_diff_mean <= 0.001): {len(category_1)} datasets")
for r in category_1:
    print(f"  - {r['dataset']}: {r['mean_diff_mean']:.6f}")

print(f"\nCategory 2 (0.001 < mean_diff_mean <= 0.1): {len(category_2)} datasets")
for r in category_2:
    print(f"  - {r['dataset']}: {r['mean_diff_mean']:.6f}")

print(f"\nCategory 3 (mean_diff_mean > 0.1): {len(category_3)} datasets")
for r in category_3:
    print(f"  - {r['dataset']}: {r['mean_diff_mean']:.6f}")

print("\n" + "=" * 60)
print("LaTeX Table Output")
print("=" * 60)
print(latex_output)

# Save LaTeX to file
with open('eval_outputs_gemini3flash_260129/dataset_classification.tex', 'w') as f:
    f.write(latex_output)

print(f"\nLaTeX table saved to: eval_outputs_gemini3flash_260129/dataset_classification.tex")

# Generate Markdown table
def generate_markdown_table(category_1, category_2, category_3):
    """Generate a Markdown table for dataset classification"""
    md_output = []
    md_output.append("# Dataset Classification by Mean Difference (Gemini 3 Flash)\n")
    
    # Summary
    md_output.append("## Summary\n")
    md_output.append(f"- **Category 1 (Excellent)**: {len(category_1)} datasets ($\\bar{{\\epsilon}} \\leq 0.001$)")
    md_output.append(f"- **Category 2 (Good)**: {len(category_2)} datasets ($0.001 < \\bar{{\\epsilon}} \\leq 0.1$)")
    md_output.append(f"- **Category 3 (Poor)**: {len(category_3)} datasets ($\\bar{{\\epsilon}} > 0.1$)\n")
    
    # Category 1
    md_output.append("## Category 1: Excellent ($\\bar{\\epsilon} \\leq 0.001$)\n")
    md_output.append("| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |")
    md_output.append("|---------|-----------|----------|---------|------------|")
    for r in category_1:
        md_output.append(f"| {r['dataset']} | {format_scientific(r['mean_diff_mean'])} | {format_scientific(r['max_diff_mean'])} | {r['tc_mean']:.3f} | {r['max_iter']} |")
    md_output.append("")
    
    # Category 2
    md_output.append("## Category 2: Good ($0.001 < \\bar{\\epsilon} \\leq 0.1$)\n")
    md_output.append("| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |")
    md_output.append("|---------|-----------|----------|---------|------------|")
    for r in category_2:
        md_output.append(f"| {r['dataset']} | {format_scientific(r['mean_diff_mean'])} | {format_scientific(r['max_diff_mean'])} | {r['tc_mean']:.3f} | {r['max_iter']} |")
    md_output.append("")
    
    # Category 3
    md_output.append("## Category 3: Poor ($\\bar{\\epsilon} > 0.1$)\n")
    md_output.append("| Dataset | Mean Diff | Max Diff | TC Mean | Max Iter |")
    md_output.append("|---------|-----------|----------|---------|------------|")
    for r in category_3:
        md_output.append(f"| {r['dataset']} | {format_scientific(r['mean_diff_mean'])} | {format_scientific(r['max_diff_mean'])} | {r['tc_mean']:.3f} | {r['max_iter']} |")
    
    return "\n".join(md_output)

# Generate and save Markdown table
markdown_output = generate_markdown_table(category_1, category_2, category_3)

print("\n" + "=" * 60)
print("Markdown Table Output")
print("=" * 60)
print(markdown_output)

with open('eval_outputs_gemini3flash_260129/dataset_classification.md', 'w') as f:
    f.write(markdown_output)

print(f"\nMarkdown table saved to: eval_outputs_gemini3flash_260129/dataset_classification.md")
