import numpy as np
import os
from math import *
from src.HybridAutomata import HybridAutomata
from HA_evaluation import TrajectoryPlotter
import json


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
            # plot data using TrajectoryPlotter
            system_title = os.path.splitext(os.path.basename(json_path))[0]
            title_suffix = f" Sample {cnt - 1}" if cnt is not None else ""
            plot_title = f"{system_title} {title_suffix} - Time Series".strip()
            figure_path = os.path.join(data_path, f"sample_{state_id}.png")
            
            plotter = TrajectoryPlotter(
                state_data=state_data,
                input_data=input_data,
                dt=dT,
                input_plot=True,
                title=plot_title
            )
            plotter.plot(mode="single")
            plotter.save(figure_path)
            plotter.close()

            # save the data
            np.savez(os.path.join(data_path, "sample_" + str(state_id)),
                     state=state_data, mode=mode_data, input=input_data, change_points=change_points)
            state_id += 1


if __name__ == "__main__":
    creat_data('automata/non_linear/duffing.json', 'data_duffing', 0.001, 10)
    # creat_data('automata/ATVA/ball.json', 'data_ball', 0.001, 10)
    # creat_data('automata/non_linear/lander.json', 'data_lander', 0.01, 10)
