import numpy as np

def save_results(trainer, dir, env=None):
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

    arrays_dict = {}
    trainer.workers.foreach_env(reset_env)
    arrays_dict["rewards"] = np.array([reward for reward in trainer.workers.foreach_env(get_rewards) if reward != []])
    arrays_dict["areas"] = np.array([area for area in trainer.workers.foreach_env(get_areas) if area != []])
    arrays_dict["delays"] = np.array([delay for delay in trainer.workers.foreach_env(get_delays) if delay != []])
    arrays_dict["actions"] = np.array([action for action in trainer.workers.foreach_env(get_actions) if action != []])

    # if trainer.evaluation_config["evaluation_interval"] is not None:
    trainer.evaluate()

    arrays_dict["eval_rewards"] = np.array([reward for reward in trainer.evaluation_workers.foreach_env(get_rewards) if reward != []])
    arrays_dict["eval_areas"] = np.array([area for area in trainer.evaluation_workers.foreach_env(get_areas) if area != []])
    arrays_dict["eval_delays"] = np.array([delay for delay in trainer.evaluation_workers.foreach_env(get_delays) if delay != []])
    arrays_dict["eval_actions"] = np.array([action for action in trainer.evaluation_workers.foreach_env(get_actions) if action != []])

    if env is not None:
        obs = env.reset()
        state = trainer.workers.local_worker().get_policy().get_initial_state()
        for i in range(env.env_config["horizon"]):
            action, state, extra = trainer.compute_single_action(observation=obs, state=state, explore=False)
            obs, reward, done, info = env.step(action)
        env.reset()
        arrays_dict["greedy_rewards"] = np.array(env.logger.rewards)
        arrays_dict["greedy_areas"] = np.array(env.logger.areas)
        arrays_dict["greedy_delays"] = np.array(env.logger.delays)
        arrays_dict["greedy_actions"] = np.array(env.logger.actions)

    np.savez(dir, **arrays_dict)


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
        if self.env.step_num==self.env.horizon:    # only save and reset if the episode had any steps
            self.rewards.append(self.current_rewards.copy())
            self.areas.append(self.current_areas.copy())
            self.delays.append(self.current_delays.copy())
            self.actions.append(self.current_actions.copy())
        self.current_rewards = []
        self.current_areas = []
        self.current_delays = []
        self.current_actions = []

