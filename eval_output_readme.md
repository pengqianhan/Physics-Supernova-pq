# Evaluation Output Readme

## main hyperparameters:
    MODEL="gemini/gemini-3-flash-preview"
    MAX_ITERATIONS=50
    FEEDBACK_TOP_K=50
    TARGET_ERROR=0.001
    NO_IMPROVEMENT_PATIENCE=20

## Dataset:
data_all: run file: utils/Dainarx_code/CreatData.py creat_all_data()
          dt = 0.001, total_time = 10.0
## Spend money:
API cost: about 108 USD
## Training data:
in 'load_trace_data_from_filepath' there are codes like 'for sample_id in sample_ids[:3]:', which means only the first 3 samples are used for training.

## 

