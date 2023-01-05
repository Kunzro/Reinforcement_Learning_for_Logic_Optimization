import argparse
import os

import yaml
# from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig

from aig_env import Abc_Env, Mockturtle_Env
from utils import MyCallbacks, logger_creator
from logger import save_results
from models import GCN

parser = argparse.ArgumentParser(description="Train a RL agent to optimize logic circuits.")
parser.add_argument('--config_file', help="specify the config file used to run the experiment. eg from ./configs", required=True)
args = parser.parse_args()

with open(args.config_file, 'r') as file:
    experiment_config = yaml.safe_load(file)
    assert isinstance(experiment_config, dict), "yaml loading failed!"

if experiment_config["env"] == "Abc_Env":
    experiment_env = Abc_Env
elif experiment_config["env"] == "Mockturtle_Env":
    experiment_env = Mockturtle_Env
else:
    raise Exception("invalid train env selected")

conf = PPOConfig()
conf = conf.training(
    sgd_minibatch_size=80,
    train_batch_size = 800,
    model={
        "custom_model": GCN,
        "custom_model_config": experiment_config
    }
)
conf = conf.environment(
    env=experiment_env,
    env_config=experiment_config
)

conf = conf.framework(framework='torch')
conf = conf.rollouts(
    num_rollout_workers=20,
    rollout_fragment_length=experiment_config["MAX_STEPS"],
    batch_mode='complete_episodes',
    horizon=experiment_config["MAX_STEPS"],
    preprocessor_pref=None
)
conf = conf.reporting(keep_per_episode_custom_metrics=True)
conf = conf.debugging(
    logger_creator=logger_creator,
    logger_config=experiment_config,
    log_level='WARN'
)
conf = conf.callbacks(callbacks_class=MyCallbacks)
conf = conf.resources(num_gpus=1)
conf = conf.experimental(_disable_preprocessor_api=True)
ppo_algo = conf.build()

for i in range(experiment_config["train_iterations"]):
    result = ppo_algo.train()

# save stats and the used config
results_dir = os.path.join(ppo_algo.logdir, "results.npz")
save_results(ppo_algo, results_dir)

config_dir = os.path.join(ppo_algo.logdir, "experiment_config.yml")
with open(config_dir, 'w') as file:
    yaml.safe_dump(experiment_config, file, default_flow_style=False)

del ppo_algo
