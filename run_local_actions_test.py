import argparse
from datetime import datetime
import os
import numpy as np
import yaml

from aig_env import AbcLocalOperations
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
    

    experiment_config["horizon"] = experiment_config["sgd_minibatch_size"]*experiment_config["train_iterations"]
    env = AbcLocalOperations(env_config=experiment_config)
    model = LocalOpsModel(env.num_node_features, env.num_actions).to(device)
    total_rewards = []
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    gamma = 0.99
    obs = env.reset()
    total_reward = 0

    # create logdir
    date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
    logdir_prefix = "{}_{}_{}_{}_{}".format(experiment_config["experiment_name"], experiment_config["circuit_name"], experiment_config["env"], experiment_config["algorithm"], date_str)
    logdir = os.path.join(os.getcwd(), "results", logdir_prefix)
    os.makedirs(logdir, exist_ok=False)
    writer = SummaryWriter(log_dir=logdir)
    writer.add_scalar("total_reward", total_reward, env.step_num)
    writer.add_scalar("area", env.area, env.step_num)
    writer.add_scalar("delay", env.delay, env.step_num)

    train_iterator = tqdm(range(experiment_config["train_iterations"]), position=1)
    train_iterator.set_description("train_iterations")
    for i in train_iterator:
        rewards = []
        probs = []
        values = []
        graph_data = []
        states  = []
        graph_data.append(obs["graph_data"])
        states.append(obs["states"])

        for j in range(experiment_config["sgd_minibatch_size"]):
            # get the action probabilities
            prob = model(obs["states"].to(device), obs["graph_data"].to(device)).flatten()
            pi = Categorical(probs=prob)
            action = pi.sample()

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            probs.append(prob[action])
            values.append(model.value_function())
            total_reward += reward
            writer.add_scalar("total_reward", total_reward, env.step_num)
            writer.add_scalar("area", env.area, env.step_num)
            writer.add_scalar("delay", env.delay, env.step_num)

            if done:
                break

        total_rewards.append(sum(rewards))

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
    logs["areas"] = env.logger.areas
    logs["delays"] = env.logger.delays
    logs["rewards"] = env.logger.rewards
    logs["actions"] = env.logger.actions
    logs["node_ids"] = env.logger.node_ids
    np.savez(os.path.join(logdir, "results.npz"), **logs)

    config_dir = os.path.join(logdir, "experiment_config.yml")
    with open(config_dir, 'w') as file:
        yaml.safe_dump(experiment_config, file, default_flow_style=False)
