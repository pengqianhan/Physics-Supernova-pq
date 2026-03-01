#!/bin/bash
#
# evaluate_all.sh — Evaluate all HA JSON specs against ground truth data.
#
# For each JSON in utils/Dainarx_code/automata/<benchmark>/<system>.json,
# evaluates against ground_truth_{0,1,2}.npz and averages metrics.
# Simulations start from mode 1.
#
# Usage:
#   bash evaluate_all.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " HA Evaluation: All automata × ground_truth_{0,1,2}"
echo "============================================================"
echo ""

python3 evaluate_all.py
