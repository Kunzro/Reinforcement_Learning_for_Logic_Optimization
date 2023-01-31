import os
import time
import tracemalloc
from ctypes import byref, c_float, c_int

import numpy as np
import torch
import yaml
from gym import Env, spaces
from ray.rllib.utils import check_env
from ray.rllib.utils.spaces.repeated import Repeated
from torch_geometric.data import Data

from abc_ctypes import (Abc_FrameGetGlobalFrame, Abc_RLfLOBalanceNode,
                        Abc_RLfLOGetEdges_wrapper,
                        Abc_RLfLOGetMaxDelayTotalArea,
                        Abc_RLfLOGetNodeFeatures_wrapper,
                        Abc_RLfLOGetNumNodesAndLevels,
                        Abc_RLfLOMapGetAreaDelay_wrapper, Abc_RLfLONtkDesub,
                        Abc_RLfLONtkRefactor, Abc_RLfLONtkResubstitute,
                        Abc_RLfLONtkRewrite, Abc_Start, Abc_Stop,
                        Cmd_CommandExecute)
from extern.RLfLO_mockturtle.build import Mockturtle_api
from logger import RLfLO_logger
from utils import onehot_encode


class Mockturtle_Env(Env):

    def __init__(self, env_config):
        self.env_config = env_config
        self.graph_type = env_config['mockturtle']['graph_type']
        self.use_graph = env_config["use_graph"]
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.target_delay = env_config["target_delay"]
        self.horizon = self.env_config["horizon"]
        self._max_episode_steps = self.env_config["horizon"]

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
                    "node_types": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=2000),
                    "edge_index": Repeated(spaces.Box(0, 100000, shape=(2,), dtype=np.int32), max_len=4000),
                    "edge_attr": Repeated(spaces.Box(0, 10, shape=(1,), dtype=np.int32), max_len=4000)
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
        elif self.step_num >= self.horizon:
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


class Abc_Env(Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config, verbose=False, trmalloc=False) -> None:
        self.verbose = verbose
        self.trmalloc = trmalloc
        self.env_config = env_config
        self.use_graph = env_config["use_graph"]
        self.use_previous_action = env_config["use_previous_action"]
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.target_delay = env_config["target_delay"]
        self.horizon = self.env_config["horizon"]
        self._max_episode_steps = self.env_config["horizon"]
        self.num_actions = len(env_config["optimizations"]["aig"])

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

        Abc_Start()
        self.pAbc = Abc_FrameGetGlobalFrame()
        library_dir = os.path.join(os.getcwd(), self.env_config["library_file"])
        Cmd_CommandExecute(self.pAbc, ('read_lib -v ' + library_dir).encode('UTF-8'))   # load library file
        self.reset()

        # define action and observation spaces
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict(
            {
                "states": spaces.Box(low=0, high=10000000, shape=(4,), dtype=np.float32),
            }
        )
        if self.use_previous_action:
            self.observation_space["previous_action"] = spaces.Box(low=0, high=len(env_config["optimizations"]["mig"])+1, shape=(1,), dtype=np.int32)
        if self.use_graph:
            self.observation_space["node_features"] = spaces.Box(0, 10, shape=(self.max_nodes, 5), dtype=np.float32)
            self.observation_space["edge_index"] = spaces.Box(0, 100000, shape=(2, self.max_edges), dtype=np.int64)
            self.observation_space["edge_attr"] = spaces.Box(0, 10, shape=(self.max_edges, 1), dtype=np.float32)
            self.observation_space["node_data_size"] = spaces.Box(1, self.max_nodes, shape=(1,), dtype=np.int32)
            self.observation_space["edge_data_size"] = spaces.Box(1, self.max_edges, shape=(1,), dtype=np.int32)

    
    def _get_obs(self, reset=False):
        """get the observation consisting of delay area numNodes and numLevels after mapping"""
        # save the previous metrics
        if not reset:
            self.prev_delay = self.delay
            self.prev_area = self.area
            self.prev_num_nodes = self.num_nodes
            self.prev_num_levels = self.num_levels
        # Cmd_CommandExecute(self.pAbc, b'strash')

        obs = {}

        if self.use_graph:
            types, num_inv = self._get_node_features()
            edge_index, edge_attr = self._get_edge_index()
            # onehot encode types
            node_features = np.concatenate((onehot_encode(types, max=3), num_inv[..., np.newaxis]), axis=1)
            if not hasattr(self, 'max_nodes') and not hasattr(self, 'max_edges'): # set max num nodes and edges to 3x the initial num
                self.max_nodes = int(node_features.shape[0]*4)  # 3 worked for log2
                self.max_edges = int(edge_attr.shape[0]*4)
            node_data_size = node_features.shape[0]
            edge_data_size =  edge_index.shape[1]
            assert self.max_nodes-node_data_size >= 0, "the observation {} is bigger than the maximum size of the array {}.".format(node_data_size, self.max_nodes)
            assert self.max_edges-edge_data_size >= 0, "the observation {} is bigger than the maximum size of the array {}.".format(edge_data_size, self.max_edges)
            assert edge_data_size == edge_attr.shape[0], "the size for edge attr and edge index should be the same." # make sure egde_attr size == edge_data size
            obs["node_features"] = np.pad(node_features, ((0, self.max_nodes-node_data_size), (0, 0)))
            obs["edge_index"] = np.pad(edge_index, ((0,0), (0, self.max_edges-edge_data_size)))
            obs["edge_attr"] = np.pad(edge_attr, ((0, self.max_edges-edge_data_size), (0, 0)))
            obs["node_data_size"] = np.array(node_data_size, dtype=np.int32).reshape((1,))
            obs["edge_data_size"] = np.array(edge_data_size, dtype=np.int32).reshape((1,))

        if self.use_previous_action:
            if self.action is not None:
                obs["previous_action"] = np.array(self.action, dtype=np.int32).reshape((1,))
            else:
                obs["previous_action"] = np.array(self.num_actions, dtype=np.int32).reshape((1,))

        if self.trmalloc:
            obs_snapshot_start = tracemalloc.take_snapshot() 
        Abc_RLfLOGetNumNodesAndLevels(self.pAbc, byref(self.c_num_nodes), byref(self.c_num_levels))             # get numNodes and numLevels
        if "use_builtin_map" in self.env_config and self.env_config["use_builtin_map"]:
            Cmd_CommandExecute(self.pAbc, (f"map -D {self.target_delay}").encode('UTF-8'))
            Abc_RLfLOGetMaxDelayTotalArea(self.pAbc, byref(self.c_delay), byref(self.c_area), 0, 0)
            Cmd_CommandExecute(self.pAbc, b'strash; strash')
        else:
            # Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 0, 0, 0, 0, 0)             # map and get area and delay DEFAULT MODE
            # Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 1, 0, 0, 0, 0, 0)           # map and get area and delay AREA ONLY MODE
            Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 1, self.target_delay, 0, 0, 0)    # map and get area and delay TARGET DELAY MODE
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

        obs["states"] = np.array([self.delay, self.area, self.num_nodes, self.num_levels], dtype=np.float32)/self.stats_normalization

        return obs

    def _get_info(self):
        return {}

    def _get_node_features(self):
        types, num_inv = Abc_RLfLOGetNodeFeatures_wrapper(pAbc=self.pAbc)
        return types, num_inv

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
        self.action = None # None = env was resetted

        # load the circuit and get the initial observation
        circuit_dir = os.path.join(os.getcwd(), self.env_config["circuit_file"])
        Cmd_CommandExecute(self.pAbc, ('read ' + circuit_dir).encode('UTF-8'))               # load circuit
        Cmd_CommandExecute(self.pAbc, b'strash')
        Abc_RLfLOGetNumNodesAndLevels(self.pAbc, byref(self.c_num_nodes), byref(self.c_num_levels))             # get numNodes and numLevels
        # Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 0, 0, 0, 0, 0)             # map and get area and delay DEFAULT MODE
        # Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 1, 0, 0, 0, 0, 0)           # map and get area and delay AREA ONLY MODE
        Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 1, self.target_delay, 0, 0, 0)    # map and get area and delay TARGET DELAY MODE
        self.delay = self.c_delay.value
        self.area = self.c_area.value
        self.num_nodes = self.c_num_nodes.value
        self.num_levels = self.c_num_levels.value

        # save initial metrics
        self.initial_delay = self.delay
        self.initial_area = self.area
        self.initial_num_nodes = self.num_nodes
        self.initial_num_levels = self.num_levels

        self.stats_normalization = np.array([self.target_delay, self.initial_area, self.initial_num_nodes, self.initial_num_levels], dtype=np.float32)

        obs = self._get_obs(reset=True)
        info = self._get_info()
    
        return obs


    def step(self, action):
        self.step_num += 1
        self.action = action
        if self.done:
            raise Exception("An Environment that is done shouldn't call step()!")
        elif self.step_num >= self.horizon:
            self.done = True

        # apply the action selected by the actor
        strash_start = time.time()
        if self.trmalloc:
            strash_snapshot_start = tracemalloc.take_snapshot()
        # Cmd_CommandExecute(self.pAbc, b'strash')
        if self.trmalloc:
            strash_snapshot_end = tracemalloc.take_snapshot()
        strash_end = time.time()
        # print(f"RUNNING: prev actions: {self.logger.current_actions}, current action: {action}")
        Cmd_CommandExecute(self.pAbc, self.env_config['optimizations']['aig'][action].encode('UTF-8'))
        # print(f"COMPLETED: prev actions: {self.logger.current_actions}, current action: {action}")
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

        self.logger.log_step()

        return obs, reward, self.done, info


    def close(self):
        Abc_Stop()


class AbcLocalOperations():

    def __init__(self, env_config: dict):
        self.env_config = env_config
        self.num_actions = len(env_config["optimizations"])
        self.target_delay = env_config["target_delay"]
        self.delay_reward_factor = env_config["delay_reward_factor"]
        self.horizon = self.env_config["horizon"]

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

        Abc_Start()
        self.pAbc = Abc_FrameGetGlobalFrame()
        library_dir = os.path.join(os.getcwd(), self.env_config["library_file"])
        Cmd_CommandExecute(self.pAbc, ('read_lib -v ' + library_dir).encode('UTF-8'))   # load library file
        self.reset()

    def reset(self):
        """reset the state of Abc by simply reloading the original circuit"""
        self.logger.log_episode()
        self.step_num = 0
        self.episode = 0   
        self.done = False
        self.reward = 0
        self.accumulated_reward = 0
        self.current_trajectory = []
        self.action = None # None = env was resetted

        # load the circuit and get the initial observation
        circuit_dir = os.path.join(os.getcwd(), self.env_config["circuit_file"])
        Cmd_CommandExecute(self.pAbc, ('read ' + circuit_dir).encode('UTF-8'))          # load circuit
        Cmd_CommandExecute(self.pAbc, b'strash')
        Abc_RLfLOGetNumNodesAndLevels(self.pAbc, byref(self.c_num_nodes), byref(self.c_num_levels))     # get numNodes and numLevels
        Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 1, self.target_delay, 0, 0, 0)    # map and get area and delay TARGET DELAY MODE

        self.delay = self.c_delay.value
        self.area = self.c_area.value
        self.num_nodes = self.c_num_nodes.value
        self.num_levels = self.c_num_levels.value

        # save initial metrics
        self.initial_delay = self.delay
        self.initial_area = self.area
        self.initial_num_nodes = self.num_nodes
        self.initial_num_levels = self.num_levels

        self.stats_normalization = torch.tensor([self.target_delay, self.initial_area, self.initial_num_nodes, self.initial_num_levels], dtype=torch.float32)

        obs = self._get_obs(reset=True)
        info = self._get_info()
        if self.done:
            raise Exception("An Environment that is done shouldn't call step()!")
        elif self.step_num >= self.horizon:
            self.done = True

        self.num_node_features = obs["graph_data"].num_node_features

        return obs, info, 

    def step(self, action):
        
        # find the action and node id from the "global action"
        node_id = action // self.num_actions
        action = action % self.num_actions

        if self.done:
            raise Exception("An Environment that is done shouldn't call step()!")
        elif self.step_num >= self.horizon:
            self.done = True
    
        self.step_num += 1
        self.action = action

        action_str = self.env_config['optimizations'][action]

        if action_str == "rewrite": # fUpdateLevels = 1
            Abc_RLfLONtkRewrite(self.pAbc, node_id, 1, 0, 0, 0, 0)
        elif action_str == "rewrite -z":
            Abc_RLfLONtkRewrite(self.pAbc, node_id, 1, 1, 0, 0, 0)
        elif action_str == "refactor": # nodeSizeMax 10 ConeSizeMax 16 fUpdateLevel =  1
            Abc_RLfLONtkRefactor(self.pAbc, node_id, 10, 16, 1, 0, 0, 0)
        elif action_str == "refactor -z":
            Abc_RLfLONtkRefactor(self.pAbc, node_id, 10, 16, 1, 1, 0, 0)
        elif action_str == "resub": # nCutsMax     =  8; nNodesMax    =  1; nLevelsOdc   =  0; fUpdateLevel =  1;
            Abc_RLfLONtkResubstitute(self.pAbc, node_id, 8, 1, 0, 1, 0, 0)
        # elif action_str == "resub -z": Use Zero doesn't exist!?
        #     Abc_RLfLONtkResubstitute(self.pAbc, node_id, 8, 1, 0, 1, 0, 0)
        elif action_str == "balance":
            Abc_RLfLOBalanceNode(self.pAbc, node_id, 1, 0)
        elif action_str == "balance -d":
            Abc_RLfLOBalanceNode(self.pAbc, node_id, 1, 1)
        elif action_str == "desub":
            Abc_RLfLONtkDesub(self.pAbc, node_id)
        else:
            raise Exception("Invalid action!")

        obs = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        self.reward = reward

        self.logger.log_step()

        return obs, reward, self.done, info

    def _get_obs(self, reset=False):
        # save the previous metrics
        if not reset:
            self.prev_delay = self.delay
            self.prev_area = self.area
            self.prev_num_nodes = self.num_nodes
            self.prev_num_levels = self.num_levels
        
        obs = {}
        
        types, num_invs = Abc_RLfLOGetNodeFeatures_wrapper(self.pAbc) # node types aren't one hot encoded yet!
        edge_index, edge_attr = Abc_RLfLOGetEdges_wrapper(self.pAbc)
        delays = Abc_RLfLOMapGetAreaDelay_wrapper(self.pAbc, self.c_area, self.c_delay, 0, 1, self.target_delay, 0, 0, 1)

        # calculate slack from delays and normalize by target_delay
        slacks = torch.from_numpy((delays - self.target_delay)/self.target_delay).unsqueeze(1)
        num_invs = torch.from_numpy(num_invs).unsqueeze(1)
        types = torch.from_numpy(onehot_encode(types, max=3))
        node_features = torch.cat((types, num_invs, slacks), dim=1)
        # node_features = torch.tensor((types, num_invs, slacks), dtype=torch.float).reshape((types.shape[0], -1)) # todo make sure shape is correct
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        obs["graph_data"] = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        self.delay = self.c_delay.value
        self.area = self.c_area.value
        self.num_nodes = self.c_num_nodes.value
        self.num_levels = self.c_num_levels.value

        obs["states"] = torch.tensor([self.delay, self.area, self.num_nodes, self.num_levels], dtype=torch.float32)/self.stats_normalization

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

    def _get_info(self):
        return {}

    def __del__(self):
        Abc_Stop()


if __name__ == "__main__":
    adder_config = os.path.abspath("configs/square.yml")
    with open(adder_config, 'r') as file:
        env_config = yaml.safe_load(file)
    
    test_aig_env = True

    if test_aig_env:
        env = Abc_Env(env_config=env_config)
        env.reset()
        check_env(env)
        env.reset()
        env = Abc_Env(env_config=env_config)
        env.reset()
        for j in range(10):
            num_nodes = []
            areas = []
            delays = []
            for i in range(40):
                res = env.step(env.action_space.sample())
                #print(res[2])
                num_nodes.append(res[0]["states"][2])
                areas.append(res[0]["states"][1])
                delays.append(res[0]["states"][0])
            env.reset()
            num_nodes = np.array(num_nodes)
            num_nodes_sorted = np.sort(num_nodes)[::-1]
            #print(num_nodes)
            print("areas:")
            print(areas)
            print()
            print("delay:")
            print(delays)
            print()
            print(np.all(num_nodes == num_nodes_sorted))
        print("wait")

    else:    
        env = Mockturtle_Env(env_config=env_config)
        env.reset()
        print("checking env")
        check_env(env=env)
        print("check successfull")

        env = Mockturtle_Env(env_config=env_config)
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
