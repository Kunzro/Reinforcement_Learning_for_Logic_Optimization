from ctypes import byref, c_float, c_int
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np
import yaml
from gym import Env, spaces
from gym.utils.env_checker import check_env
from logger import RLfLO_logger
import time
import os
import tracemalloc
from extern.RLfLO_mockturtle.build import Mockturtle_api

from abc_ctypes import (Abc_FrameGetGlobalFrame, Abc_RLfLOGetEdges_wrapper, Abc_RLfLOGetMaxDelayTotalArea,
                        Abc_RLfLOGetNumNodesAndLevels, Abc_RLfLOGetObjTypes_wrapper, Abc_Start, Abc_Stop,
                        Cmd_CommandExecute)

class Mockturtle_env(Env):

    def __init__(self, env_config):
        self.env_config = env_config
        self.graph_type = env_config['mockturtle']['graph_type']
        self.use_graph = env_config["use_graph"]
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.target_delay = env_config["target_delay"]
        self.MAX_STEPS = self.env_config["MAX_STEPS"]
        self._max_episode_steps = self.env_config["MAX_STEPS"]

        # keep trac of the history and best episode/trajectory
        self.logger = RLfLO_logger(self)
        self.step_num = 0
        self.episode = 0   
        self.done = False
        self.reward = 0
        self.accumulated_reward = 0
        self.current_trajectory = []

        # define action and observation spaces
        self.action_space = spaces.Discrete(len(env_config["optimizations"][self.graph_type]))
        if self.use_graph:
            self.observation_space = spaces.Dict(
                {
                    "states": spaces.Box(low=0, high=10000000, shape=(4,), dtype=np.float32),
                    "node_types": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=10000),
                    "edge_index": Repeated(spaces.Box(0, 100000, shape=(2,), dtype=np.int32), max_len=10000),
                    "edge_attr": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=10000)
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "states": spaces.Box(low=0, high=10000000, shape=(4,), dtype=np.float64),
                }
            )

        # create mtl object
        self.genlib_dir = os.path.abspath(env_config['library_file'])
        self.circuit_dir = os.path.abspath(env_config["circuit_file"])

    def _get_info(self):
        return {}

    def _get_obs(self, reset=False):
        if not reset:
            self.prev_delay = self.delay
            self.prev_area = self.area
            self.prev_num_nodes = self.num_nodes
            self.prev_num_levels = self.num_levels

        # IMPORTANT map before getting stats
        self.mtl.map(self.genlib_dir)

        self.delay = self.mtl.get_delay()
        self.area = self.mtl.get_area()
        self.num_nodes = self.mtl.get_size()
        self.num_levels = self.mtl.get_depth()

        obs = {}
        obs["states"] = np.array([self.area, self.delay, self.num_nodes, self.num_levels], np.float32)

        return obs

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
        

    def step(self, action):
        self.step_num += 1
        if self.done:
            raise Exception("An Environment that is done shouldn't call step()!")
        elif self.step_num >= self.MAX_STEPS:
            self.done = True

        action_str = self.env_config['optimizations'][self.graph_type][action]

        # apply the action selected by the actor
        if action_str == 'rewrite':
            self.mtl.rewrite(False, False, False, 3)
        elif action_str == 'rewrite -azg':
            self.mtl.rewrite(True, False, False, 3)
        elif action_str == 'rewrite -udc':
            self.mtl.rewrite(False, True, False, 3)
        elif action_str == 'rewrite -pd':
            self.mtl.rewrite(False, False, True, 3)
        elif action_str == 'rewrite -azg -udc':
            self.mtl.rewrite(True, True, False, 3)
        elif action_str == 'rewrite -azg -pd':
            self.mtl.rewrite(True, False, True, 3)
        elif action_str == 'rewrite -udc -pd':
            self.mtl.rewrite(False, True, True, 3)
        elif action_str == 'rewrite -azg -udc -pd':
            self.mtl.rewrite(True, True, True, 3)
        elif action_str == 'balance':
            self.mtl.balance(False, 4)
        elif action_str == 'balance -c':
            self.mtl.balance(True, 4)
        elif action_str == 'refactor': # might be broken
            self.mtl.refactor(False, False)
        elif action_str == 'refactor -azg': # might be broken
            self.mtl.refactor(True, False)
        elif action_str == 'refactor -udc': # might be broken
            self.mtl.refactor(False, True)
        elif action_str == 'refactor -azg -udc': # might be broken
            self.mtl.refactor(True, True)
        elif action_str == 'resub':
            self.mtl.resub(8, 2, False, 12, False)
        elif action_str == 'resub -pd':
            self.mtl.resub(8, 2, False, 12, True)
        else:
            raise Exception("Invalid action!")
        
        # get the new observation, info and reward
        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()

        self.reward = reward
        self.accumulated_reward += reward
        self.action = action

        self.logger.log_step()

        return obs, reward, self.done, info

    def reset(self):
        """reset the state of Abc by simply reloading the original circuit"""
        self.logger.log_episode()
        self.step_num = 0
        self.episode = 0   
        self.done = False
        self.reward = 0
        self.accumulated_reward = 0
        self.current_trajectory = []

        # load the circuit and get the initial observation
        self.mtl = Mockturtle_api.Mockturtle_mig_api()
        self.mtl.load_verilog(self.circuit_dir)
        obs = self._get_obs(reset=True)

        # save initial metrics
        self.initial_delay = self.delay
        self.initial_area = self.area
        self.initial_num_nodes = self.num_nodes
        self.initial_num_levels = self.num_levels
    
        return obs


class Aig_Env(Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config, verbose=False, trmalloc=False) -> None:
        self.verbose = verbose
        self.trmalloc = trmalloc
        self.env_config = env_config
        self.use_graph = env_config["use_graph"]
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.target_delay = env_config["target_delay"]
        self.MAX_STEPS = self.env_config["MAX_STEPS"]
        self._max_episode_steps = self.env_config["MAX_STEPS"]

        # tracemalloc if needed
        if self.trmalloc:
            tracemalloc.start()

        # keep trac of the history and best episode/trajectory
        self.logger = RLfLO_logger(self)
        self.step_num = 0
        self.episode = 0   
        self.done = False
        self.reward = 0
        self.accumulated_reward = 0
        self.current_trajectory = []

        # initialize variables to be used to get metrices from ABC
        self.c_delay = c_float()
        self.c_area = c_float()
        self.c_num_nodes = c_int()
        self.c_num_levels = c_int()

        # define action and observation spaces
        self.action_space = spaces.Discrete(len(env_config["optimizations"]["aig"]))
        if self.use_graph:
            self.observation_space = spaces.Dict(
                {
                    "states": spaces.Box(low=0, high=10000000, shape=(4,), dtype=np.float32),
                    "node_types": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=10000),
                    "edge_index": Repeated(spaces.Box(0, 100000, shape=(2,), dtype=np.int32), max_len=10000),
                    "edge_attr": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=10000)
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "states": spaces.Box(low=0, high=10000000, shape=(4,), dtype=np.float32),
                }
            )

        Abc_Start()
        self.pAbc = Abc_FrameGetGlobalFrame()
        library_dir = os.path.join(os.getcwd(), self.env_config["library_file"])
        Cmd_CommandExecute(self.pAbc, ('read ' + library_dir).encode('UTF-8'))   # load library file

    
    def _get_obs(self, reset=False):
        """get the observation consisting of delay area numNodes and numLevels after mapping"""
        # save the previous metrics
        if not reset:
            self.prev_delay = self.delay
            self.prev_area = self.area
            self.prev_num_nodes = self.num_nodes
            self.prev_num_levels = self.num_levels
        Cmd_CommandExecute(self.pAbc, b"strash")

        obs = {}

        if self.use_graph:
            node_types = self._get_node_types()
            edge_index, edge_attr = self._get_edge_index()
            obs["node_types"] = node_types
            obs["edge_index"] = edge_index
            obs["edge_attr"] = edge_attr

        if self.trmalloc:
            obs_snapshot_start = tracemalloc.take_snapshot() 
        Cmd_CommandExecute(self.pAbc, b'map')                                                                   # map
        Abc_RLfLOGetNumNodesAndLevels(self.pAbc, byref(self.c_num_nodes), byref(self.c_num_levels))             # get numNodes and numLevels
        Abc_RLfLOGetMaxDelayTotalArea(self.pAbc, byref(self.c_delay), byref(self.c_area), 0, 0, 0, 0, 0)        # get size and delay
        if self.trmalloc:
            obs_snapshot_end = tracemalloc.take_snapshot()
            obs_stats = obs_snapshot_end.compare_to(obs_snapshot_start, 'lineno')
            print('\n' * 2, "OBS STATS:")
            for stat in obs_stats[:10]:
                print(stat)
        self.delay = self.c_delay.value
        self.area = self.c_area.value
        self.num_nodes = self.c_num_nodes.value
        self.num_levels = self.c_num_levels.value

        obs["states"] = np.array([self.delay, self.area, self.num_nodes, self.num_levels], dtype=np.float32)

        return obs

    def _get_info(self):
        return {}

    def _get_node_types(self):
        node_types = Abc_RLfLOGetObjTypes_wrapper(pAbc=self.pAbc)
        return node_types

    def _get_edge_index(self):
        edge_index, edge_attr = Abc_RLfLOGetEdges_wrapper(pAbc=self.pAbc)
        return edge_index, edge_attr

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


    def reset(self):
        """reset the state of Abc by simply reloading the original circuit"""
        self.logger.log_episode()
        self.step_num = 0
        self.episode = 0   
        self.done = False
        self.reward = 0
        self.accumulated_reward = 0
        self.current_trajectory = []

        # load the circuit and get the initial observation
        circuit_dir = os.path.join(os.getcwd(), self.env_config["circuit_file"])
        Cmd_CommandExecute(self.pAbc, ('read ' + circuit_dir).encode('UTF-8'))               # load circuit
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
        strash_start = time.time()
        if self.trmalloc:
            strash_snapshot_start = tracemalloc.take_snapshot()
        Cmd_CommandExecute(self.pAbc, b'strash')
        if self.trmalloc:
            strash_snapshot_end = tracemalloc.take_snapshot()
        strash_end = time.time()
        Cmd_CommandExecute(self.pAbc, self.env_config['optimizations']['aig'][action].encode('UTF-8'))
        if self.trmalloc:
            command_snapshot_end = tracemalloc.take_snapshot()
            strash_stats = strash_snapshot_end.compare_to(strash_snapshot_start, 'lineno')
            command_stats = command_snapshot_end.compare_to(strash_snapshot_end, 'lineno')
            print('\n' * 2, "STRASH STATS:")
            for stat in strash_stats[:10]:
                print(stat)
            print('\n' * 2, "COMMAND STATS:")
            for stat in command_stats[:10]:
                print(stat)
        command_end = time.time()

        # get the new observation, info and reward
        obs_start = time.time()
        obs = self._get_obs()
        obs_end = time.time()
        info = self._get_info()
        reward = self._get_reward()
        if self.verbose:
            print(f"Strash time: {strash_end-strash_start:.5f}, Command time: {command_end-strash_end:.5f}, Obs time: {obs_end-obs_start:.5f}")

        self.reward = reward
        self.action = action

        self.logger.log_step()

        return obs, reward, self.done, info


    def close(self):
        Abc_Stop()


if __name__ == "__main__":
    adder_config = os.path.abspath("configs/multiplier.yml")
    with open(adder_config, 'r') as file:
        env_config = yaml.safe_load(file)
    
    test_aig_env = False

    if test_aig_env:
        env = Aig_Env(env_config=env_config)
        env.reset()
        check_env(env=env)
        env = Aig_Env(env_config=env_config)
        env.reset()
        for j in range(10):
            for i in range(50):
                res = env.step(env.action_space.sample())
                print(res[2])
            env.reset()
        print("wait")

    else:    
        env = Mockturtle_env(env_config=env_config, graph_type='mig')
        env.reset()
        print("checking env")
        check_env(env=env)
        print("check successfull")

        env = Mockturtle_env(env_config=env_config, graph_type='mig')
        env.reset()

        print("test all actions")
        for i in range(len(env_config["optimizations"]['mig'])):
            env.step(i)
        env.reset()
        print("all actions tested")
        
        num_rollouts = 10
        print(f"testing {num_rollouts} random rollouts of length {env_config['train_iterations']}")
        for j in range(10):
            for i in range(env_config['train_iterations']):
                action = env.action_space.sample()
                print(f"applying action: {env_config['optimizations']['mig'][action]}")
                res = env.step(action=action)
            print(f"rollout {j} complete")
            env.reset()

    print("test ended")
