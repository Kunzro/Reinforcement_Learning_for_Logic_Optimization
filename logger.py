import numpy as np
from ray.rllib.agents.trainer import Trainer

def save_results(trainer: Trainer, dir):
    def get_rewards(env):
        return env.logger.rewards
    def get_areas(env):
        return env.logger.areas
    def get_delays(env):
        return env.logger.delays
    def get_actions(env):
        return env.logger.actions
    def reset_env(env):
        env.reset()

    trainer.workers.foreach_env(reset_env)
    rewards = np.array([reward for reward in trainer.workers.foreach_env(get_rewards) if reward != []])
    areas = np.array([area for area in trainer.workers.foreach_env(get_areas) if area != []])
    delays = np.array([delay for delay in trainer.workers.foreach_env(get_delays) if delay != []])
    actions = np.array([action for action in trainer.workers.foreach_env(get_actions) if action != []])

    np.savez(dir, rewards=rewards, areas=areas, delays=delays, actions=actions)


def load_results(dir):
    results = np.load(dir, allow_pickle=False)
    return results


class RLfLO_logger():
    def __init__(self, env):
        self.env = env
        self.num_steps = 0
        self.rewards = []           # list of lists tracking the rewards at each step per episode
        self.areas = []             # list of lists tracking the areas at each step per episode
        self.delays = []            # list of lists tracking the delays at each step per episode$
        self.actions = []           # list of lists tracking the actions at each step per episode
        self.current_rewards = []
        self.current_areas = []
        self.current_delays = []
        self.current_actions = []

    def log_step(self):
        """call this at every step of the episode"""
        self.current_rewards.append(self.env.reward)
        self.current_areas.append(self.env.area)
        self.current_delays.append(self.env.delay)
        self.current_actions.append(self.env.action)

    def log_episode(self):
        """call this at the end of the episode to log the episode eg in the rest function of the env"""
        if self.env.step_num==self.env.MAX_STEPS:    # only save and reset if the episode had any steps
            self.rewards.append(self.current_rewards.copy())
            self.areas.append(self.current_areas.copy())
            self.delays.append(self.current_delays.copy())
            self.actions.append(self.current_actions.copy())
        self.current_rewards = []
        self.current_areas = []
        self.current_delays = []
        self.current_actions = []

