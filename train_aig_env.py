import numpy as np
from aig_env import Aig_Env
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks
import ray.rllib.agents.ppo as ppo
import yaml
from datetime import datetime
import os
from logger import save_results
import argparse


parser = argparse.ArgumentParser(description="Train a RL agent to optimize logic circuits.")
parser.add_argument('--config_file', help="specify the config file used to run the experiment. eg from ./configs", required=True)
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    experiment_config = yaml.safe_load(file)

config = ppo.DEFAULT_CONFIG.copy()
config["env_config"] = experiment_config
config["framework"] = "torch"
config["env"] = Aig_Env
config["num_gpus"] = 0
config["num_workers"] = 5
config["batch_mode"] = "complete_episodes"
config["num_gpus_per_worker"] = 0
config["num_envs_per_worker"] = 1
config["rollout_fragment_length"] = experiment_config["MAX_STEPS"]
#config["train_batch_size"] = experiment_config["MAX_STEPS"]*4
config["keep_per_episode_custom_metrics"] = True
config["preprocessor_pref"] = experiment_config["preprocessor_pref"]
#config["sgd_minibatch_size"] = experiment_config["MAX_STEPS"]

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.custom_metrics["areas"] = []
        episode.custom_metrics["delays"] = []
        episode.custom_metrics["num_nodes"] = []
        episode.custom_metrics["num_levels"] = []

    def on_episode_step(self, *, worker, base_env, policies = None, episode, **kwargs):
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        sub_envs = base_env.get_sub_environments()
        episode.custom_metrics["areas"].extend([env.area for env in sub_envs])
        episode.custom_metrics["delays"].extend([env.delay for env in sub_envs])
        episode.custom_metrics["num_nodes"].extend([env.num_nodes for env in sub_envs])
        episode.custom_metrics["num_levels"].extend([env.num_levels for env in sub_envs])
     
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        areas = np.array(result["custom_metrics"]["areas"])
        delays = np.array(result["custom_metrics"]["delays"])
        result["custom_metrics"]["area_min"] = np.min(areas)
        result["custom_metrics"]["min_area_mean"] = np.mean(np.min(areas, axis=1))
        result["custom_metrics"]["delay_at_area_min"] = delays.flatten()[np.argmin(areas)]
        result["custom_metrics"]["area_min_step_number"] = np.argmin(areas)%areas.shape[1]+1
        result["custom_metrics"]["area_min_step_hist"] = np.argmin(areas, axis=1)

config["callbacks"] = MyCallbacks
        

def logger_creator(config):
    date_str = datetime.today().strftime("%Y-%m-%d")
    logdir_prefix = "{}_{}_{}_{}".format("AIG", experiment_config["circuit_name"], "PPO", date_str)
    home_dir = os.getcwd()
    logdir = os.path.join(home_dir, "results", logdir_prefix)
    os.makedirs(logdir, exist_ok=True)
    return UnifiedLogger(config, logdir, loggers=None)

algo = PPOTrainer(config=config, logger_creator=logger_creator)

def func(env):
    print(env.area)

for i in range(experiment_config["train_iterations"]):
    result = algo.train()

# save stats and the used config
results_dir = os.path.join(algo.logdir, "results.npz")
save_results(algo, results_dir)

config_dir = os.path.join(algo.logdir, "experiment_config.yml")
with open(config_dir, 'w') as file:
    yaml.safe_dump(experiment_config, file, default_flow_style=False)

del algo