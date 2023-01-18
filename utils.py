import os
from datetime import datetime

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import UnifiedLogger


def logger_creator(config):
    logger_config = config["logger_config"]
    date_str = datetime.today().strftime("%Y-%m-%d-%H-%M")
    logdir_prefix = "{}_{}_{}_{}_{}".format(logger_config["experiment_name"], logger_config["circuit_name"], logger_config["env"], logger_config["algorithm"], date_str)
    logdir = os.path.join(os.getcwd(), "results", logdir_prefix)
    os.makedirs(logdir, exist_ok=False)
    return UnifiedLogger(config, logdir, loggers=None)


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.custom_metrics["target_delay"] = worker.config.env_config["target_delay"]
        episode.custom_metrics["areas"] = []
        episode.custom_metrics["delays"] = []
        episode.custom_metrics["num_nodes"] = []
        episode.custom_metrics["num_levels"] = []
        episode.custom_metrics["rewards"] = []

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
        episode.custom_metrics["rewards"].append(episode.total_reward)
     
    def on_train_result(self, *, result, **kwargs) -> None:
        target_delay = result["custom_metrics"]["target_delay"][0]
        areas = np.array(result["custom_metrics"]["areas"])
        delays = np.array(result["custom_metrics"]["delays"])
        rewards = np.array(result["custom_metrics"]["rewards"])
        result["custom_metrics"]["area_min"] = np.min(areas)
        result["custom_metrics"]["min_area_mean"] = np.mean(np.min(areas, axis=1))
        result["custom_metrics"]["delay_at_area_min"] = delays.flatten()[np.argmin(areas)]
        result["custom_metrics"]["area_min_step_number"] = np.argmin(areas)%areas.shape[1]+1
        result["custom_metrics"]["area_min_step_hist"] = np.argmin(areas, axis=1)
        result["custom_metrics"]["max_reward"] = rewards.max()
        result["custom_metrics"]["area_at_max_reward"] = areas.flatten()[np.argmax(rewards)]
        result["custom_metrics"]["delay_at_max_reward"] = delays.flatten()[np.argmax(rewards)]
        result["custom_metrics"]["mean_reward"] = rewards.mean()
        result["custom_metrics"]["min_reward"] = rewards.min()

        if np.any(delays<=target_delay):
            min_valid_area = areas[delays<=target_delay].min()
            delay_at_min_valid_area = delays[areas==min_valid_area].min()
            result["custom_metrics"]["min_valid_area"] = min_valid_area
            result["custom_metrics"]["delay_at_min_valid_area"] = delay_at_min_valid_area

