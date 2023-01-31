import argparse
from datetime import datetime
import os

import yaml

from aig_env import Abc_Env, Mockturtle_Env, AbcLocalOperations
from logger import save_results
from models import LocalOpsModel
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch

parser = argparse.ArgumentParser(description="Train a RL agent to optimize logic circuits.")
parser.add_argument('--config_file', help="specify the config file used to run the experiment. eg from ./configs", required=True)
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    experiment_configs = yaml.safe_load(file)
    assert isinstance(experiment_configs, dict), "yaml loading failed!"

if experiment_configs["env"] == "Abc_Env":
    experiment_env = Abc_Env
elif experiment_configs["env"] == "Mockturtle_Env":
    experiment_env = Mockturtle_Env
else:
    raise Exception("invalid train env selected")

for circuit in experiment_configs["circuits"]:

    experiment_config = experiment_configs.copy()
    experiment_config["optimizations"] = experiment_configs["optimizations"].copy()
    del experiment_config["circuits"]
    del experiment_config["target_delays"]
    del experiment_config["circuit_files"]
    experiment_config["circuit_file"] = experiment_configs["circuit_files"][circuit]
    experiment_config["circuit_name"] = circuit
    experiment_config["target_delay"] = experiment_configs["target_delays"][circuit]
    

    env = AbcLocalOperations(env_config=experiment_config)
    model = LocalOpsModel(env.num_node_features, env.num_actions)
    total_rewards = []
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    gamma = 0.99

    for i in range(experiment_config["train_iterations"]):
        rewards = []
        probs = []
        values = []
        graph_data = []
        states  = []
        obs, _ = env.reset()
        graph_data.append(obs["graph_data"])
        states.append(obs["states"])

        for i in range(experiment_config["horizon"]):
            # get the action probabilities
            prob = model(obs["states"], obs["graph_data"]).flatten()
            pi = Categorical(probs=prob)
            action = pi.sample()

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)
            probs.append(prob[action])
            values.append(model.value_function())

            if done:
                break

        total_rewards.append(sum(rewards))

        actor_losses = []
        critic_losses = []
        if done:
            R = 0
        else:
            R = values[-1]
            
        for t in range(len(rewards)-1)[::-1]:
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

    # create logdir
    date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
    logdir_prefix = "{}_{}_{}_{}_{}".format(experiment_config["experiment_name"], experiment_config["circuit_name"], experiment_config["env"], experiment_config["algorithm"], date_str)
    logdir = os.path.join(os.getcwd(), "results", logdir_prefix)
    os.makedirs(logdir, exist_ok=False)

    # save stats and the used config
    # env = experiment_env(experiment_config)
    # results_dir = os.path.join(algo.logdir, "results.npz")
    # save_results(algo, results_dir, env)

    config_dir = os.path.join(algo.logdir, "experiment_config.yml")
    with open(config_dir, 'w') as file:
        yaml.safe_dump(experiment_config, file, default_flow_style=False)
