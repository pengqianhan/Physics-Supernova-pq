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

The rest number of ground truth files are used for evaluation. ground_truth_3.npz to ground_truth_14.npz, which is 12 files.
## Validation data:
Use the first ground truth file for validation.
in run_llm_ha_gamma.py, the code is like this:
```python
npz_file_path = os.path.join(test_data_base_path, test_data_files[0])
        evaluator = HAEvaluator(
            ha_dict=ha_specification,
            npz_file_path=npz_file_path,
            dt=ha_specification['config'].get('dt', 0.001),
            total_time=ha_specification['config'].get('total_time', 10.0)
        )
```

## thinking level 

model = LiteLLMModel(
        model_id=model_id,
        api_key=os.environ.get("GEMINI_API_KEY"),
        max_completion_tokens=24576,
        num_retries=3,
        timeout=1200,
        thinking_level = "low" # high, low
    )
