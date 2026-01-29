main hyperparameters:
    MODEL="gemini/gemini-3-flash-preview"
    MAX_ITERATIONS=50
    FEEDBACK_TOP_K=50
    TARGET_ERROR=0.001
    NO_IMPROVEMENT_PATIENCE=20

data_all: run file: utils/Dainarx_code/CreatData.py creat_all_data()
          dt = 0.001, total_time = 10.0
API cost: about 108 USD


