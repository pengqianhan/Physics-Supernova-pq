import numpy as np
import os
from math import *
import matplotlib.pyplot as plt
from typing import Optional
from Dainarx_code.src.HybridAutomata import HybridAutomata
import json


def plot_fun(state_data: np.ndarray,
             input_data: np.ndarray,
             dt: float,
             system_name: str = "System",
             sample_index: Optional[int] = None,
             save_path: Optional[str] = None,
             show: bool = True) -> None:
    """Plot the time series for states/inputs."""

    if state_data.ndim != 2:
        raise ValueError("state_data must be a 2D array with shape (num_states, num_steps)")

    num_states, num_steps = state_data.shape

    if input_data.size == 0:
        input_series = np.empty((0, num_steps))
    elif input_data.ndim == 1:
        input_series = input_data.reshape(1, -1)
    elif input_data.ndim == 2:
        input_series = input_data
    else:
        raise ValueError("input_data must be empty, 1D, or 2D array")

    num_inputs = input_series.shape[0]

    if num_steps == 0:
        raise ValueError("state_data must contain at least one time step")

    time = np.arange(num_steps) * dt

    fig, ax_ts = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)

    total_series = num_states + num_inputs
    try:
        cmap = plt.cm.get_cmap('tab20', max(total_series, 1))
    except (AttributeError, TypeError):
        # For matplotlib >= 3.7, get_cmap only takes colormap name
        import matplotlib as mpl
        cmap = mpl.colormaps['tab20']

    for idx in range(num_states):
        ax_ts.plot(time, state_data[idx], label=f"x{idx + 1}", color=cmap(idx), linewidth=2)

    for idx in range(num_inputs):
        series_index = num_states + idx
        ax_ts.plot(time, input_series[idx], label=f"u{idx + 1}", linestyle='--', color=cmap(series_index), linewidth=2)

    ax_ts.set_xlabel('Time (s)')
    ax_ts.set_ylabel('Values')
    title_suffix = f" Sample {sample_index - 1}" if sample_index is not None else ""
    ax_ts.set_title(f"{system_name} {title_suffix} - Time Series".strip())
    ax_ts.grid(True, linestyle='--', alpha=0.4)
    ax_ts.legend(loc='best')

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def creat_data(json_path: str, data_path: str, dT: float, times: float):
    r"""
    :param json_path: File path of automata.
    :param data_path: Data storage path.
    :param dT: Discrete time.
    :param times: Total sampling time.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        files = os.listdir(data_path)
        for file in files:
            os.remove(os.path.join(data_path, file))

    with open(json_path, 'r') as f:
        data = json.load(f)
        sys = HybridAutomata.from_json(data['automaton'])
        state_id = 0
        cnt = 0
        for init_state in data['init_state']:
            cnt += 1
            state_data = []
            mode_data = []
            input_data = []
            change_points = [0]
            sys.reset(init_state)
            now = 0.
            idx = 0
            while now < times:
                now += dT
                idx += 1
                state, mode, switched = sys.next(dT)
                state_data.append(state)
                mode_data.append(mode)
                input_data.append(sys.getInput())
                if switched:
                    change_points.append(idx)
            change_points.append(idx)
            state_data = np.transpose(np.array(state_data))
            input_data = np.transpose(np.array(input_data))
            mode_data = np.array(mode_data)
            # print("state_data.shape: ", state_data.shape) #state_data.shape:  (1, 1001)
            # print("mode_data.shape: ", mode_data.shape)
            # print("input_data.shape: ", input_data.shape)
            # print("change_points.shape: ", len(change_points))
            '''
            ball
            state_data.shape:  (2, 1001)
            mode_data.shape:  (1001,)
            input_data.shape:  (0, 1001)
            change_points.shape:  22
            ----------------------------
            duffing
            state_data.shape:  (1, 1001)
            mode_data.shape:  (1001,)
            input_data.shape:  (1, 1001)
            change_points.shape:  13
            ----------------------------
            '''
            # plot data
            system_title = os.path.splitext(os.path.basename(json_path))[0]
            figure_path = os.path.join(data_path, f"sample_{state_id}.png")
            plot_fun(state_data, input_data, dT, system_name=system_title,
                     sample_index=cnt, save_path=figure_path, show=False)

            # save the data
            np.savez(os.path.join(data_path, "test_data" + str(state_id)),
                     state=state_data, mode=mode_data, input=input_data, change_points=change_points)
            state_id += 1


if __name__ == "__main__":
    # creat_data('automata/non_linear/duffing.json', 'data_duffing', 0.001, 10)
    # creat_data('automata/ATVA/ball.json', 'data_ball', 0.001, 10)
    creat_data('automata/non_linear/lander.json', 'data_lander', 0.01, 10)
