#!/bin/bash

# run_all.sh - Run all datasets in data_all directory
# Runs 4 datasets in parallel to respect Gemini API rate limits

# Configuration
MAX_PARALLEL=4
DATA_BASE="data_all"
MODEL="openai/kimi-k2.5"                    # Main agent model (LiteLLM-routed)
IMAGE_TOOL_MODEL="openai/kimi-k2.5"    # Image tool model (Google genai SDK, not LiteLLM)
SUMMARY_MODEL="openai/kimi-k2.5"            # Summary model (LiteLLM-routed)
MAX_ITERATIONS=5
FEEDBACK_TOP_K=5
TARGET_ERROR=0.005
NO_IMPROVEMENT_PATIENCE=20

# Log directory
LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# All datasets
DATASETS=(
    # "ATVA/ball"
    "non_linear/duffing"
)

# Function to run a single dataset
run_dataset() {
    local dataset=$1
    local log_file="$LOG_DIR/${dataset//\//_}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $dataset"

    python run_llm_ha_gamma.py \
        --input-data-path "$DATA_BASE/$dataset" \
        --manager-model "$MODEL" \
        --tools-list hybrid_automaton_image_analysis \
        --image-tool-model "$IMAGE_TOOL_MODEL" \
        --summary-model "$SUMMARY_MODEL" \
        --managed-agents-list data_analysis_expert\
        --managed-agents-list-model "$MODEL" \
        --max-iterations $MAX_ITERATIONS \
        --feedback-top-k $FEEDBACK_TOP_K \
        --target-error $TARGET_ERROR \
        --no-improvement-patience $NO_IMPROVEMENT_PATIENCE \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $dataset (success)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $dataset (exit code: $exit_code)"
    fi

    return $exit_code
}

# Export function and variables for parallel execution
export -f run_dataset
export LOG_DIR DATA_BASE MODEL IMAGE_TOOL_MODEL SUMMARY_MODEL MAX_ITERATIONS FEEDBACK_TOP_K TARGET_ERROR NO_IMPROVEMENT_PATIENCE

echo "========================================"
echo "Running all datasets with $MAX_PARALLEL parallel jobs"
echo "Log directory: $LOG_DIR"
echo "Total datasets: ${#DATASETS[@]}"
echo "========================================"
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job control"
    printf '%s\n' "${DATASETS[@]}" | parallel -j $MAX_PARALLEL run_dataset {}
else
    echo "Using bash job control (GNU parallel not found)"

    # Job counter
    running_jobs=0

    for dataset in "${DATASETS[@]}"; do
        # Wait if we have too many jobs running
        while [ $running_jobs -ge $MAX_PARALLEL ]; do
            wait -n 2>/dev/null || true
            running_jobs=$((running_jobs - 1))
        done

        # Start new job in background
        run_dataset "$dataset" &
        running_jobs=$((running_jobs + 1))
    done

    # Wait for all remaining jobs
    wait
fi

echo ""
echo "========================================"
echo "All datasets completed!"
echo "Check logs in: $LOG_DIR"
echo "========================================"
