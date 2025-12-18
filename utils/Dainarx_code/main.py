import json
import os
import re
import time
import logging
import re
import numpy as np
import matplotlib.pyplot as plt

from CreatData import creat_data
from src.DEConfig import FeatureExtractor
from src.utils import *

from src.CurveSlice import Slice, slice_curve
from src.ChangePoints import find_change_point
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state
from src.Evaluation import Evaluation
from src.HybridAutomata import HybridAutomata


def run(data_list, input_data, config, evaluation: Evaluation):
    # print("len(data_list): ", len(data_list)) #9
    # print("len(input_data): ", len(input_data)) #9 
    input_data = np.array(input_data)#shape: (9, 1, 1000)
    # print('len(data_list[0]): ', len(data_list[0]))#var_num=1
    # print('len(input_data[0]): ', len(input_data[0]))#input_num=1
    # print('input_data.shape: ', input_data.shape)#(9,1,1000)
    get_feature = FeatureExtractor(len(data_list[0]), len(input_data[0]),
                                   order=config['order'], dt=config['dt'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'])
    Slice.clear()
    slice_data = []
    chp_list = []
    for data, input_val in zip(data_list, input_data):
        change_points = find_change_point(data, input_val, get_feature, w=config['window_size'])
        chp_list.append(change_points)
        print("ChP:\t", change_points)
        slice_curve(slice_data, data, input_val, change_points, get_feature)
    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")
    Slice.Method = config['clustering_method']
    Slice.fit_threshold(slice_data)
    clustering(slice_data, config['self_loop'])
    evaluation.recording_time("clustering")
    adj = guard_learning(slice_data, get_feature, config)
    evaluation.recording_time("guard_learning")
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    evaluation.submit(slice_data=slice_data)
    return sys, slice_data


def get_config(json_path, evaluation: Evaluation):
    logging.basicConfig(level=logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    default_config = {'dt': 0.01, 'total_time': 10, 'order': 3, 'window_size': 10, 'clustering_method': 'fit',
                      'minus': False, 'need_bias': True, 'other_items': '', 'kernel': 'linear', 'svm_c': 1e6,
                      'class_weight': 1.0, 'need_reset': False, 'self_loop': False}
    config = {}
    if json_path.isspace() or json_path == '':
        config = default_config
    else:
        with open(json_path) as f:
            json_file = json.load(f)
            evaluation.submit(gt_mode_num=len(json_file.get('automaton', {'mode': []})['mode']))
            json_config = json_file.get('config', {})
            for (key, val) in default_config.items():
                if key in json_config.keys():
                    config[key] = json_config.pop(key)
                else:
                    config[key] = val
            if len(json_config) != 0:
                raise Exception('Invalid parameter: ' + str(json_config))
            f.close()
    return config, get_hash_code(json_file, config)


def main(json_path: str, data_path='data', need_creat=None, need_plot=True):
    
    evaluation = Evaluation(json_path)# evaluation.name=json_path
    config, hash_code = get_config(json_path, evaluation)
    HybridAutomata.LoopWarning = not config['self_loop']
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    if need_creat is None:
        need_creat = check_data_update(hash_code, data_path)
    if need_creat:
        print("Data being generated!")
        creat_data(json_path, data_path, config['dt'], config['total_time'])
        # save_hash_code(hash_code, data_path)

    mode_list = []
    data = []
    input_list = []
    gt_list = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    for root, dirs, files in os.walk(data_path):
        # print("root: ", root)#path/data
        # print("dirs: ", dirs)#[]
        # print("files: ", files)#['test_data0.npz', 'test_data1.npz',..., 'test_data14.npz']
        print("Loading data!")
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group())):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            # Skip sample_train_X.npz files as they don't contain 'mode' data
            if 'sample_train_' in file:
                continue
            npz_file = np.load(os.path.join(root, file))# e.g. test_data14.npz
            # print("npz_file.keys(): ", npz_file.keys()) # ['state', 'mode', 'input', 'change_points']
            # print("npz_file['state'].shape: ", npz_file['state'].shape) # (1, 10001)
            # print("npz_file['mode'].shape: ", npz_file['mode'].shape) # (10001, 1)
            # print("npz_file['input'].shape: ", npz_file['input'].shape) # (1, 10001)
            # print("npz_file['change_points'].shape: ", npz_file['change_points'].shape) # (23,)
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            gt_list.append(change_point_list)
            print("GT:\t", change_point_list.tolist())
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)
            input_list.append(npz_file['input'])

    test_num = 6

    print("Be running!")
    # print("len(gt_list): ", len(gt_list))#15
    # print("len(mode_list): ", len(mode_list))#15
    # print("len(input_list): ", len(input_list))#15
    # print("len(data): ", len(data))#15
    evaluation.submit(gt_chp=gt_list[test_num:])
    evaluation.submit(train_mode_list=mode_list[test_num:])
    evaluation.start()
    sys, slice_data = run(data[test_num:], input_list[test_num:], config, evaluation) #training to learn the hybrid automata
    print('sys',sys)
    
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")
    all_fit_mode, all_gt_mode = get_mode_list(slice_data, mode_list[test_num:])
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_list[:test_num]

    init_state_test = get_init_state(data_test, mode_map, mode_list_test, config['order'])
    fit_data_list, mode_data_list = [], []
    draw_index = 0  # If it is None, draw all the test data
    for data, mode_list, input_list, init_state in zip(data_test, mode_list_test, input_list_test, init_state_test):
        fit_data = [data[:, i] for i in range(config['order'])]## 取前0:order-1个时间步作为初始值
        mode_data = list(mode_list[:config['order']])## 取0:order-1个时间步作为初始值
        sys.reset(init_state, input_list[:, :config['order']])
        ## init_state 是初始状态(有mode和`x0`两个key)，input_list[:, :config['order']] 是前order个时间步的输入
        # simulation to test the hybrid automata
        for i in range(config['order'], data.shape[1]):## 从order个时间步开始模拟
            state, mode, switched = sys.next(input_list[:, i]) ## 基于当前输入预测下一状态
            fit_data.append(state)
            mode_data.append(mode_map_inv.get(mode, -mode))
        fit_data = np.array(fit_data)
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)
        # print('fit_data.shape: ', fit_data.shape) # (10001, 1)
        # print('data.shape: ', data.shape) # (1, 10001)
        if need_plot and (draw_index == 0 or draw_index is None):
            need_plot = not need_plot
            for var_idx in range(data.shape[0]):
                plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
                plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
                plt.show()
                plt.savefig(os.path.join(result_path, "plot_data.png"))
            plt.plot(np.arange(len(mode_list)), mode_list, color='c')
            plt.plot(np.arange(len(mode_data)), mode_data, color='r')
            plt.show()
            plt.savefig(os.path.join(result_path, "plot_mode.png"))
        if draw_index is not None:
            draw_index -= 1
    evaluation.submit(fit_mode=mode_data_list, fit_data=np.array(fit_data_list),
                      gt_mode=mode_list_test, gt_data=data_test, dt=config['dt'])
    return evaluation.calc()

def main_from_dict(ha_dict: dict, data_path='data', need_creat=None, need_plot=True):
    
    # Create a name for evaluation based on dict hash
    eval_name = f"ha_dict_{hash(json.dumps(ha_dict, sort_keys=True)) % 10000}"
    evaluation = Evaluation(eval_name)
    config, hash_code = get_config_from_dict(ha_dict, evaluation)
    HybridAutomata.LoopWarning = not config['self_loop']
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    if need_creat is None:
        need_creat = check_data_update(hash_code, data_path)
    if need_creat:
        print("Data being generated!")
        creat_data(json_path, data_path, config['dt'], config['total_time'])
        # save_hash_code(hash_code, data_path)

    mode_list = []
    data = []
    input_list = []
    gt_list = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    for root, dirs, files in os.walk(data_path):
        # print("root: ", root)#path/data
        # print("dirs: ", dirs)#[]
        # print("files: ", files)#['test_data0.npz', 'test_data1.npz',..., 'test_data14.npz']
        print("Loading data!")
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group())):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            # Skip sample_train_X.npz files as they don't contain 'mode' data
            if 'sample_train_' in file:
                continue
            npz_file = np.load(os.path.join(root, file))# e.g. test_data14.npz
            # print("npz_file.keys(): ", npz_file.keys()) # ['state', 'mode', 'input', 'change_points']
            # print("npz_file['state'].shape: ", npz_file['state'].shape) # (1, 10001)
            # print("npz_file['mode'].shape: ", npz_file['mode'].shape) # (10001, 1)
            # print("npz_file['input'].shape: ", npz_file['input'].shape) # (1, 10001)
            # print("npz_file['change_points'].shape: ", npz_file['change_points'].shape) # (23,)
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            gt_list.append(change_point_list)
            print("GT:\t", change_point_list.tolist())
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)
            input_list.append(npz_file['input'])

    test_num = 6

    print("Be running!")
    # print("len(gt_list): ", len(gt_list))#15
    # print("len(mode_list): ", len(mode_list))#15
    # print("len(input_list): ", len(input_list))#15
    # print("len(data): ", len(data))#15
    evaluation.submit(gt_chp=gt_list[test_num:])
    evaluation.submit(train_mode_list=mode_list[test_num:])
    evaluation.start()
    sys, slice_data = run(data[test_num:], input_list[test_num:], config, evaluation) #training to learn the hybrid automata
    print('sys',sys)
    
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")
    all_fit_mode, all_gt_mode = get_mode_list(slice_data, mode_list[test_num:])
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_list[:test_num]

    init_state_test = get_init_state(data_test, mode_map, mode_list_test, config['order'])
    fit_data_list, mode_data_list = [], []
    draw_index = 0  # If it is None, draw all the test data
    for data, mode_list, input_list, init_state in zip(data_test, mode_list_test, input_list_test, init_state_test):
        fit_data = [data[:, i] for i in range(config['order'])]## 取前0:order-1个时间步作为初始值
        mode_data = list(mode_list[:config['order']])## 取0:order-1个时间步作为初始值
        sys.reset(init_state, input_list[:, :config['order']])
        ## init_state 是初始状态(有mode和`x0`两个key)，input_list[:, :config['order']] 是前order个时间步的输入
        # simulation to test the hybrid automata
        for i in range(config['order'], data.shape[1]):## 从order个时间步开始模拟
            state, mode, switched = sys.next(input_list[:, i]) ## 基于当前输入预测下一状态
            fit_data.append(state)
            mode_data.append(mode_map_inv.get(mode, -mode))
        fit_data = np.array(fit_data)
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)
        # print('fit_data.shape: ', fit_data.shape) # (10001, 1)
        # print('data.shape: ', data.shape) # (1, 10001)
        if need_plot and (draw_index == 0 or draw_index is None):
            need_plot = not need_plot
            for var_idx in range(data.shape[0]):
                plt.plot(np.arange(len(data[var_idx])), data[var_idx], color='c')
                plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], color='r')
                plt.show()
                plt.savefig(os.path.join(result_path, "plot_data.png"))
            plt.plot(np.arange(len(mode_list)), mode_list, color='c')
            plt.plot(np.arange(len(mode_data)), mode_data, color='r')
            plt.show()
            plt.savefig(os.path.join(result_path, "plot_mode.png"))
        if draw_index is not None:
            draw_index -= 1
    evaluation.submit(fit_mode=mode_data_list, fit_data=np.array(fit_data_list),
                      gt_mode=mode_list_test, gt_data=data_test, dt=config['dt'])
    return evaluation.calc()

def get_config_from_dict(ha_dict: dict, evaluation: Evaluation):
    """
    Extract config from HA dictionary directly (instead of loading from file).

    Args:
        ha_dict: Dictionary containing 'automaton' and 'config' keys
        evaluation: Evaluation object for recording metrics

    Returns:
        Tuple of (config dict, hash code)
    """
    
    # in ha_dict, config['non_linear_items'] need be changed to config['other_items']
    import copy
    ha_load_dict = copy.deepcopy(ha_dict)

    # 定义键的映射关系: 旧键 -> 新键
    key_mapping = {
        'non_linear_items': 'other_items',
    }

    for old_key, new_key in key_mapping.items():
        if old_key in ha_load_dict['config'] and old_key != new_key:
            ha_load_dict['config'][new_key] = ha_load_dict['config'].pop(old_key)


    logging.basicConfig(level=logging.ERROR)
    # default_config = {'dt': 0.01, 'total_time': 10, 'order': 3, 'window_size': 10, 'clustering_method': 'fit',
    #                       'minus': False, 'need_bias': True, 'other_items': '', 'kernel': 'linear', 'svm_c': 1e6,
    #                       'class_weight': 1.0, 'need_reset': False, 'self_loop': False}
    default_config = {'dt': 0.001, 'total_time': 10, 'order': 2, 'window_size': 10, 'clustering_method': 'fit',
                      'minus': False, 'need_bias': True, 'other_items': '', 'kernel': 'rbf', 'svm_c': 1e6,
                      'class_weight': 1.0, 'need_reset': True, 'self_loop': False}
    config = {}

    evaluation.submit(gt_mode_num=len(ha_load_dict.get('automaton', {'mode': []})['mode']))
    json_config = ha_load_dict.get('config', {}).copy()  # Make a copy to avoid modifying original

    key_list = ['order','other_items']
    for (key, val) in default_config.items():
        if key in key_list:
            config[key] = json_config.pop(key)
        else:
            config[key] = val
    # Note: Remaining keys in json_config are intentionally ignored
    # We only extract keys specified in key_list

    #加一个正则表达式，如果other_items中包含形如x1[?]，那么转换为x[?]
    config['other_items'] = re.sub(r'x\d+\[\?\]', 'x[?]', config['other_items'])

    return config, get_hash_code(ha_load_dict, config)


def main_from_dictv0(ha_dict: dict, data_path: str = 'data', need_creat: bool = None, need_plot: bool = False):
    """
    Run Dainarx validation with HA specification passed as a dictionary.

    This function enables the ManagerAgent to:
    1. Generate an HA JSON specification dynamically
    2. Pass it directly to Dainarx for validation (no file I/O needed)
    3. Get evaluation metrics back

    Args:
        ha_dict: Dictionary containing 'automaton' and 'config' keys
        data_path: Path to directory containing .npz ground truth files
        need_creat: Whether to create new data (if None, auto-detect)
        need_plot: Whether to generate plots

    Returns:
        Dictionary containing evaluation metrics (tc, max_diff, mean_diff, etc.)
    """
    # Create a name for evaluation based on dict hash
    eval_name = f"ha_dict_{hash(json.dumps(ha_dict, sort_keys=True)) % 10000}"
    evaluation = Evaluation(eval_name)

    config, hash_code = get_config_from_dict(ha_dict, evaluation)
    HybridAutomata.LoopWarning = not config['self_loop']

    # print('config (from dict): ')
    # for key, value in config.items():
    #     print(f'\t{key}: {value}')

    # Handle data path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)

    # If need_creat, generate data from the HA specification
    if need_creat is None:
        need_creat = check_data_update(hash_code, data_path)
    if need_creat:
        print("Data being generated from HA dict!")
        # Write temp JSON file for creat_data (it requires a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(ha_dict, tmp_file)
            tmp_json_path = tmp_file.name
        try:
            creat_data(tmp_json_path, data_path, config['dt'], config['total_time'])
        finally:
            os.unlink(tmp_json_path)  # Clean up temp file

    # Load data from npz files
    mode_list = []
    data = []
    input_list = []
    gt_list = []

    print("Loading data!")
    for root, dirs, files in os.walk(data_path):
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group()) if re.search(r'(\d+)', x) else 0):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            if 'sample_train_' in file:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            gt_list.append(change_point_list)
            print("GT:\t", change_point_list.tolist())
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)
            input_list.append(npz_file['input'])

    if len(data) == 0:
        return {"error": f"No valid .npz files found in {data_path}"}

    test_num = min(6, len(data) // 2)  # Adaptive test/train split

    print("Running Dainarx validation!")
    evaluation.submit(gt_chp=gt_list[test_num:])
    evaluation.submit(train_mode_list=mode_list[test_num:])
    evaluation.start()

    # Run the learning algorithm
    sys, slice_data = run(data[test_num:], input_list[test_num:], config, evaluation)
    print('sys', sys)

    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")
    all_fit_mode, all_gt_mode = get_mode_list(slice_data, mode_list[test_num:])
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_list[:test_num]

    init_state_test = get_init_state(data_test, mode_map, mode_list_test, config['order'])
    fit_data_list, mode_data_list = [], []

    for data_item, mode_item, input_item, init_state in zip(data_test, mode_list_test, input_list_test, init_state_test):
        fit_data = [data_item[:, i] for i in range(config['order'])]
        mode_data = list(mode_item[:config['order']])
        sys.reset(init_state, input_item[:, :config['order']])

        for i in range(config['order'], data_item.shape[1]):
            state, mode, switched = sys.next(input_item[:, i])
            fit_data.append(state)
            mode_data.append(mode_map_inv.get(mode, -mode))

        fit_data = np.array(fit_data)
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)

    evaluation.submit(fit_mode=mode_data_list, fit_data=np.array(fit_data_list),
                      gt_mode=mode_list_test, gt_data=data_test, dt=config['dt'])

    return evaluation.calc()


def validate_ha_specification(ha_dict: dict, data_path: str = 'data') -> dict:
    """
    Validate an HA specification against ground truth data using Dainarx.

    This is a convenience wrapper around main_from_dict() for use by ManagerAgent.

    Args:
        ha_dict: HA specification dictionary with 'automaton' and 'config' keys
        data_path: Path to directory containing ground truth .npz files

    Returns:
        Dictionary with evaluation results including:
        - 'success': bool indicating if validation succeeded
        - 'metrics': dict with tc, max_diff, mean_diff, etc.
        - 'feedback': string feedback for the agent
        - 'ha_spec': the validated HA specification
    """
    try:
        eval_results = main_from_dict(ha_dict, data_path, need_creat=False, need_plot=False)

        if 'error' in eval_results:
            return {
                'success': False,
                'metrics': {},
                'feedback': f"Validation error: {eval_results['error']}",
                'ha_spec': ha_dict
            }

        # Extract key metrics
        metrics = {
            'tc': eval_results.get('tc'),
            'train_tc': eval_results.get('train_tc'),
            'max_diff': eval_results.get('max_diff'),
            'mean_diff': eval_results.get('mean_diff'),
            'clustering_error': eval_results.get('clustering_error'),
        }

        # Build feedback string
        feedback_parts = [
            f"HA Specification validated against ground truth data.",
            f"Results:",
            f"  - TC (Change-Point Error): {metrics['tc']:.6f}" if metrics['tc'] is not None else "  - TC: N/A",
            f"  - Max Difference: {metrics['max_diff']:.6f}" if metrics['max_diff'] is not None else "  - Max Diff: N/A",
            f"  - Mean Difference: {metrics['mean_diff']:.6f}" if metrics['mean_diff'] is not None else "  - Mean Diff: N/A",
        ]

        return {
            'success': True,
            'metrics': metrics,
            'feedback': "\n".join(feedback_parts),
            'ha_spec': ha_dict
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'metrics': {},
            'feedback': f"Validation failed with error: {str(e)}\n{traceback.format_exc()}",
            'ha_spec': ha_dict
        }


if __name__ == "__main__":
    result_path = "result"
    # check if the result folder exists if not, creat it
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    ha_dict = {
  "automaton": {
    "var": "x1",
    "input": "u1",
    "mode": [
      {
        "id": 1,
        "eq": "x1[2] = -0.1 * x1[1] - 10.0 * x1[0] - 1.0 * x1[0] ** 3 + u1"
      }
    ],
    "edge": []
  },
  "config": {
    "dt": 0.001,
    "total_time": 10.0,
    "order": 2,
    "need_reset": False,
    "non_linear_items": "x1[?] ** 3"
  }
}
    # eval_log = main_from_dictv0(ha_dict, data_path='data', need_creat=False, need_plot=False)
    eval_log = main_from_dict(ha_dict, data_path='data', need_creat=False, need_plot=False)
    # eval_log = main("./automata/non_linear/duffing.json")

    with open(os.path.join(result_path, "eval_log.json"), "w") as f:
        json.dump(eval_log, f)
    print("Evaluation log:")
    for key_, val_ in eval_log.items():
        print(f"{key_}: {val_}")
