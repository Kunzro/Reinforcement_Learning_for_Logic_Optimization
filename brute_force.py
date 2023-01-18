import argparse
from datetime import datetime
from multiprocessing import Pool, TimeoutError
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np

import yaml
from aig_env import Abc_Env


def perform_rollout(actions):
    env = Abc_Env(env_config=experiment_config)
    for action in actions:
        env.step(action)
    env.horizon = env.step_num    # make sure the results get logged
    env.logger.log_episode()
    return [env.logger.rewards[0], env.logger.areas[0], env.logger.delays[0], env.logger.actions[0]]


def save_results(results, logdir):
    save_dir = os.path.join(logdir, "results.npz")
    results = np.swapaxes(results, 0, 1)
    np.savez(save_dir, rewards=results[0], areas=results[1], delays=results[2], actions=results[3])


def trajectories_of_len(num_actions, length):
    trajectories = []
    trajectory = [0]*length
    trajectories.append(trajectory.copy())
    while trajectory != [num_actions-1]*length:
        for idx, el in enumerate(trajectory):
            if el+1 == num_actions:
                trajectory[idx] = 0
            else:
                trajectory[idx] +=1
                break
        trajectories.append(trajectory.copy())
    return trajectories

def random_trajectories(num_actions, length, num_rand):
    trajectories = np.random.randint(0, num_actions, (num_rand, length), dtype=np.int32).tolist()
    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best solution using brute force.")
    parser.add_argument('--config_file', help="specify the config file used to run the experiment. eg from ./configs", required=True)
    parser.add_argument('--seq_len', help='the max length of a sequence/rollout', required=False, type=int)
    parser.add_argument('--mode', help='the mode of brute forcing, either random or all', required=True)
    parser.add_argument('--num_rand', help='the number of random rollouts', required=False, type=int)
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        experiment_configs = yaml.safe_load(file)

    for circuit in experiment_configs["circuits"]:
        
        experiment_config = experiment_configs.copy()
        experiment_config["optimizations"] = experiment_configs["optimizations"].copy()
        del experiment_config["circuits"]
        del experiment_config["target_delays"]
        del experiment_config["circuit_files"]
        experiment_config["circuit_file"] = experiment_configs["circuit_files"][circuit]
        experiment_config["circuit_name"] = circuit
        experiment_config["target_delay"] = experiment_configs["target_delays"][circuit]
        experiment_config["mode"] = args.mode
        if "map -D ; strash" in experiment_config["optimizations"]["aig"]:
            experiment_config["optimizations"]["aig"] = experiment_config["optimizations"]["aig"].copy()
            map_index = experiment_config["optimizations"]["aig"].index("map -D ; strash")
            experiment_config["optimizations"]["aig"][map_index] = "map -D {}; strash".format(experiment_config["target_delay"])


        date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
        num_actions = len(experiment_config["optimizations"]["aig"])
        if args.mode == 'all':
            assert hasattr(args, 'seq_len'), "you have to specify the seq_len for the all mode!"
            mode_str = f"Max_Seq_len_{args.seq_len}"
            experiment_config["seq_len"] = args.seq_len
            trajectories = trajectories_of_len(num_actions, args.seq_len)
        elif args.mode == 'random':
            assert hasattr(args, 'num_rand'), "you have to specify the num_rand for the random mode!"
            mode_str = f"Num_Rand_Traj_{args.num_rand}"
            experiment_config["num_rand"] = args.num_rand
            trajectories = random_trajectories(num_actions, experiment_config['horizon'], args.num_rand)

        logdir_prefix = "{}_{}_{}_{}_{}".format(experiment_config["experiment_name"], experiment_config["circuit_name"], experiment_config["env"], mode_str, date_str)
        home_dir = os.getcwd()
        logdir = os.path.join(home_dir, "results", logdir_prefix)
        os.makedirs(logdir, exist_ok=False)

        num_processes = 40
        p = Pool(num_processes)
        res = p.map(perform_rollout, trajectories)
        p.close()
        p.join()
        results = np.array(res.get())
        save_results(results, logdir)
        del p

        config_dir = os.path.join(logdir, "experiment_config.yml")
        with open(config_dir, 'w') as file:
            yaml.safe_dump(experiment_config, file, default_flow_style=False)