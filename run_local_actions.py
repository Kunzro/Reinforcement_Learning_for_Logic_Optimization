import argparse
from datetime import datetime
import os
import numpy as np
import yaml

from aig_env import AbcLocalOperations, AbcLocalOperationsSparse
from logger import save_results
from models import LocalOpsModel
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train a RL agent to optimize logic circuits.")
parser.add_argument('--config_file', help="specify the config file used to run the experiment. eg from ./configs", required=True)
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    experiment_configs = yaml.safe_load(file)
    assert isinstance(experiment_configs, dict), "yaml loading failed!"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

circuit_iterator = tqdm(experiment_configs["circuits"], position=0)
for circuit in circuit_iterator:
    circuit_iterator.set_description(f"running circuit: {circuit}")
    experiment_config = experiment_configs.copy()
    experiment_config["optimizations"] = experiment_configs["optimizations"].copy()
    del experiment_config["circuits"]
    del experiment_config["target_delays"]
    del experiment_config["circuit_files"]
    experiment_config["circuit_file"] = experiment_configs["circuit_files"][circuit]
    experiment_config["circuit_name"] = circuit
    experiment_config["target_delay"] = experiment_configs["target_delays"][circuit]
    

    experiment_config["horizon"] = experiment_config["sgd_minibatch_size"]*max(experiment_config["train_iterations"])
    if experiment_config["sparse"]:
        env = AbcLocalOperationsSparse(env_config=experiment_config)
    else:
        env = AbcLocalOperations(env_config=experiment_config)
    model = LocalOpsModel(env.num_state_features, env.num_node_features, env.num_actions, experiment_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    gamma = 0.99
    obs = env.reset()

    # create logdir
    date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
    logdir_prefix = "{}_{}_{}_{}_{}_desub-{}_balance-d-{}_{}".format(experiment_config["global_softmax"], experiment_config["normal_reward_factor"], experiment_config["delay_reward_factor"], experiment_config["level_reward_factor"], experiment_config["num_sparse_rewards"], "desub" in experiment_config["optimizations"], "balance -d" in experiment_config["optimizations"], date_str)
    logdir = os.path.join(os.getcwd(), "results", experiment_config["experiment_name"], experiment_config["circuit_name"], logdir_prefix)
    os.makedirs(logdir, exist_ok=False)
    # writer = SummaryWriter(log_dir=logdir)

    config_dir = os.path.join(logdir, "experiment_config.yml")
    with open(config_dir, 'w') as file:
        yaml.safe_dump(experiment_config, file, default_flow_style=False)

    for num_iterations, iter_name in zip(experiment_config["train_iterations"], experiment_config["iterations_name"]):  # list of number of train iterations before env should be reset
        if iter_name is not None:
            logdir = os.path.join(os.getcwd(), "results", experiment_config["experiment_name"], experiment_config["circuit_name"], logdir_prefix, iter_name)
            os.makedirs(logdir, exist_ok=False)
        else:
            logdir = os.path.join(os.getcwd(), "results", experiment_config["experiment_name"], experiment_config["circuit_name"], logdir_prefix)
        writer = SummaryWriter(log_dir=logdir)
        total_reward = 0
        obs = env.reset()
        env.horizon = experiment_config["sgd_minibatch_size"]*num_iterations
        writer.add_scalar("total_reward", total_reward, env.step_num)
        writer.add_scalar("area", env.area, env.step_num)
        writer.add_scalar("delay", env.delay, env.step_num)
        writer.add_scalar("num_nodes", env.num_nodes, env.step_num)
        writer.add_scalar("num_levels", env.num_levels, env.step_num)
        train_iterator = tqdm(range(num_iterations), position=1)
        train_iterator.set_description("train_iterations")
        for i in train_iterator:
            rewards = []
            probs = []
            values = []
            actions = []

            for j in range(experiment_config["sgd_minibatch_size"]):
                # get the action probabilities
                prob = model(obs["states"].to(device), obs["graph_data"].to(device)).flatten()
                pi = Categorical(probs=prob)
                action = pi.sample()

                obs, reward, done, info = env.step(action.item())
                actions.append(env.action)
                rewards.append(reward)
                probs.append(prob[action])
                values.append(model.value_function())
                total_reward += reward
                writer.add_scalar("total_reward", total_reward, env.step_num)
                writer.add_scalar("cur_reward", reward, env.step_num)
                writer.add_scalar("area", env.area, env.step_num)
                writer.add_scalar("delay", env.delay, env.step_num)
                writer.add_scalar("num_nodes", env.num_nodes, env.step_num)
                writer.add_scalar("num_levels", env.num_levels, env.step_num)

                if done:
                    break

            writer.add_histogram(tag="actions_dist", values = np.array(actions, dtype=np.int32), global_step=env.step_num, bins=env.num_actions, max_bins=env.num_actions)

            actor_losses = []
            critic_losses = []
            if done:
                R = 0
            else:
                R = values[-1]
                
            for t in reversed(range(len(rewards))):
                with torch.no_grad():
                    R = rewards[t] + gamma*R
                    advantage = R - values[t]
                
                actor_losses.append(advantage*torch.log(probs[t]))
                critic_losses.append(torch.square(R-values[t]))
                
            actor_losses_tensor = torch.stack(actor_losses, 0)
            critic_losses_tensor = torch.stack(critic_losses, 0)
            
            loss = torch.sum(critic_losses_tensor) - torch.sum(actor_losses_tensor)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

    # save results
    env.reset()
    logs = {}
    for idx, name in enumerate(experiment_config["iterations_name"]):
        logs[f"{name}_areas"] = env.logger.areas[idx]
        logs[f"{name}_delays"] = env.logger.delays[idx]
        logs[f"{name}_rewards"] = env.logger.rewards[idx]
        logs[f"{name}_actions"] = env.logger.actions[idx]
        logs[f"{name}_node_ids"] = env.logger.node_ids[idx]
        logs[f"{name}_num_nodes"] = env.logger.num_nodes[idx]
        logs[f"{name}_num_levels"] = env.logger.num_levels[idx]
    logdir = os.path.join(os.getcwd(), "results", experiment_config["experiment_name"], experiment_config["circuit_name"], logdir_prefix)
    np.savez(os.path.join(logdir, "results.npz"), **logs)

