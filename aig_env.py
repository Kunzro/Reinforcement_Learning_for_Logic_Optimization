import numpy as np
import yaml

from gym import Env, spaces
from gym.utils.env_checker import check_env
from ctypes import c_float, c_int, byref
from abc_ctypes import Abc_RLfLOGetMaxDelayTotalArea, Abc_RLfLOGetNumNodesAndLevels, Abc_Start, Abc_Stop, Abc_FrameGetGlobalFrame, Cmd_CommandExecute


class Aig_Env(Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config) -> None:
        self.env_config = env_config
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.target_delay = env_config["target_delay"]
        self.MAX_STEPS = self.env_config["MAX_STEPS"]

        # initialize variables to be used to get metrices from ABC
        self.c_delay = c_float()
        self.c_area = c_float()
        self.c_num_nodes = c_int()
        self.c_num_levels = c_int()

        # define action and observation spaces
        self.action_space = spaces.Discrete(len(env_config["optimizations"]["aig"]))
        self.observation_space = spaces.Dict(
            {
                "delay_area": spaces.Box(low=0, high=1000000000, shape=(2,), dtype=float),
                "nodes_levels": spaces.Box(low=0, high=1000000000, shape=(2,), dtype=int),
            }
        )

        # Start the Abc fram TODO: check if this works with multiprocessing
        Abc_Start()
        self.pAbc = Abc_FrameGetGlobalFrame()
        Cmd_CommandExecute(self.pAbc, ('read ' + self.env_config["library_file"]).encode('UTF-8'))   # load library file

    
    def _get_obs(self, reset=False):
        """get the observation consisting of delay area numNodes and numLevels after mapping"""
        # save the previous metrics
        if not reset:
            self.prev_delay = self.delay
            self.prev_area = self.area
            self.prev_num_nodes = self.num_nodes
            self.prev_num_levels = self.num_levels
        Cmd_CommandExecute(self.pAbc, b"strash")
        Cmd_CommandExecute(self.pAbc, b'map')                                                                   # map
        Abc_RLfLOGetNumNodesAndLevels(self.pAbc, byref(self.c_num_nodes), byref(self.c_num_levels))             # get numNodes and numLevels
        Abc_RLfLOGetMaxDelayTotalArea(self.pAbc, byref(self.c_delay), byref(self.c_area), 0, 0, 0, 0, 0)        # get size and delay
        self.delay = self.c_delay.value
        self.area = self.c_area.value
        self.num_nodes = self.c_num_nodes.value
        self.num_levels = self.c_num_levels.value
        return {"delay_area": np.array([self.delay, self.area], dtype=float), "nodes_levels": np.array([self.num_nodes, self.num_levels], dtype=int)}


    def _get_info(self):
        return {}


    def _get_reward(self):
        """the reward is the improvement in area and delay (untill the delay target is met)
        scaled by the initial area and delay respectively"""
        area_reward = (self.prev_area - self.area)/self.initial_area
        if self.prev_delay > self.target_delay:
            if self.delay > self.target_delay:
                delay_reward = (self.prev_delay - self.delay)/self.target_delay
            else:
                delay_reward = (self.prev_delay - self.target_delay)/self.target_delay
        else:
            if self.delay > self.target_delay:
                delay_reward = (self.target_delay - self.delay)/self.target_delay
            else:
                delay_reward = 0

        return area_reward + self.delay_reward_factor * delay_reward


    def reset(self, seed = None, options=None):
        """reset the state of Abc by simply reloading the original circuit"""
        super().reset(seed=seed)

        self.episode = 0   
        self.step_num = 0
        self.done = False

        # load the circuit and get the initial observation
        Cmd_CommandExecute(self.pAbc, ('read ' + self.env_config["circuit_file"]).encode('UTF-8'))               # load circuit
        obs = self._get_obs(reset=True)
        info = self._get_info()

        # save initial metrics
        self.initial_delay = self.delay
        self.initial_area = self.area
        self.initial_num_nodes = self.num_nodes
        self.initial_num_levels = self.num_levels
    
        return obs


    def step(self, action):
        self.step_num += 1
        if self.done:
            raise Exception("An Environment that is done shouldn't call step()!")
        elif self.step_num >= self.MAX_STEPS:
            self.done = True
        # apply the action selected by the actor
        Cmd_CommandExecute(self.pAbc, b'strash')
        Cmd_CommandExecute(self.pAbc, self.env_config['optimizations']['aig'][action].encode('UTF-8'))
        # get the new observation, info and reward
        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()

        return obs, reward, self.done, info


    def close(self):
        Abc_Stop()


if __name__ == "__main__":
    with open('/home/roman/Documents/Studium/Masterthesis_RL_for_logic_synthesis/code-nosync/Reinforcement_Learning_for_Logic_Optimization/configs/adder.yml', 'r') as file:
        env_config = yaml.safe_load(file)
    env = Aig_Env(env_config=env_config)
    env.reset()
    check_env(env=env)