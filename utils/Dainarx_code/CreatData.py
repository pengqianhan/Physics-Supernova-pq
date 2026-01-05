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
        os.makedirs(data_path)
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
                input_plot=False,
                title=plot_title
            )
            plotter.plot(mode="single")
            plotter.save(figure_path)
            plotter.close()

            # save the data
            np.savez(os.path.join(data_path, "sample_" + str(state_id)),
                     state=state_data, mode=mode_data, input=input_data, change_points=change_points)
            np.savez(os.path.join(data_path, "sample_train_" + str(state_id)),
                     state=state_data, input=input_data)
            state_id += 1


def creat_all_data(automata_dir: str = None, output_dir: str = None, default_dt: float = 0.001, default_total_time: float = 10.0):
    r"""
    Create datasets for all JSON files under the automata directory.

    :param automata_dir: Path to the automata directory containing JSON files.
                         Defaults to 'utils/Dainarx_code/automata' relative to project root.
    :param output_dir: Path to the output directory for generated data.
                       Defaults to 'data_all' in the project root.
    :param default_dt: Default discrete time step if not specified in JSON config.
    :param default_total_time: Default total sampling time if not specified in JSON config.
    """
    # Determine project root (two levels up from this file: CreatData.py -> Dainarx_code -> utils -> root)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))

    # Set default paths
    if automata_dir is None:
        automata_dir = os.path.join(current_file_dir, 'automata')
    elif not os.path.isabs(automata_dir):
        automata_dir = os.path.join(project_root, automata_dir)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'data_all')
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all JSON files under automata directory
    json_files = []
    for root, dirs, files in os.walk(automata_dir):
        for file in files:
            if file.endswith('.json') and file != 'json_readme.md':
                json_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} JSON files in {automata_dir}")

    # Process each JSON file
    for json_path in sorted(json_files):
        # Get relative path from automata_dir
        rel_path = os.path.relpath(json_path, automata_dir)
        # Remove .json extension and use as directory name
        rel_dir = os.path.splitext(rel_path)[0]
        # Create corresponding output path
        data_output_path = os.path.join(output_dir, rel_dir)

        # Read JSON to get config (dt and total_time)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Get dt and total_time from config, use defaults if not present
            config = data.get('config', {})
            dt = config.get('dt', default_dt)
            total_time = config.get('total_time', default_total_time)

            print(f"\nProcessing: {rel_path}")
            print(f"  Output: {os.path.relpath(data_output_path, project_root)}")
            print(f"  dt={dt}, total_time={total_time}")

            # Create data using creat_data function
            creat_data(json_path, data_output_path, dt, total_time)
            print(f"  Done!")

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")
            continue

    print(f"\nAll data created in: {output_dir}")


if __name__ == "__main__":
    # Example: create data for a single automaton
    creat_data('automata/non_linear/duffing.json', 'data_duffing', 0.001, 10)
    # creat_data('automata/ATVA/ball.json', 'data_ball', 0.001, 10)
    # creat_data('automata/non_linear/lander.json', 'data_lander', 0.01, 10)

    # Create data for all automata
    # creat_all_data()
