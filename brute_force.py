import argparse
from datetime import datetime
from multiprocessing import Pool
import os
import numpy as np

import yaml
from aig_env import Abc_Env


def perform_rollout(actions):
    env = Abc_Env(env_config=experiment_config)
    for action in actions:
        env.step(action)
    env.MAX_STEPS = env.step_num    # make sure the results get logged
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
        experiment_config = yaml.safe_load(file)

    date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
    logdir_prefix = "{}_{}_{}_{}".format(experiment_config["env"], experiment_config["circuit_name"], "BF", date_str)
    if "special_tag" in experiment_config:
        logdir_prefix += "_{}".format(experiment_config["special_tag"])

    num_actions = len(experiment_config["optimizations"]["aig"])
    if args.mode == 'all':
        assert hasattr(args, 'seq_len'), "you have to specify the seq_len for the all mode!"
        logdir_prefix += f"_all_{args.seq_len}"
        trajectories = trajectories_of_len(num_actions, args.seq_len)
    elif args.mode == 'random':
        assert hasattr(args, 'num_rand'), "you have to specify the num_rand for the random mode!"
        logdir_prefix += f"_random_{args.num_rand}"
        trajectories = random_trajectories(num_actions, experiment_config['MAX_STEPS'], args.num_rand)

    home_dir = os.getcwd()
    logdir = os.path.join(home_dir, "results", logdir_prefix)
    os.makedirs(logdir, exist_ok=False)

    p = Pool(36)
    res = p.map_async(perform_rollout, trajectories)
    p.close()
    p.join()
    results = np.array(res.get())
    save_results(results, logdir)

    config_dir = os.path.join(logdir, "experiment_config.yml")
    with open(config_dir, 'w') as file:
        yaml.safe_dump(experiment_config, file, default_flow_style=False)