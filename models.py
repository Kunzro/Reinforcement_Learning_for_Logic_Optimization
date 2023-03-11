import torch
import gym
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GINEConv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from torch_geometric.data import Data, Batch

class LocalOpsModel(torch.nn.Module):
    # IMPORTANT DOES NOT SUPPORT MINIBATCHES YET
    def __init__(self, num_state_features, num_node_features, num_actions, config):
        super().__init__()
        self.config = config
        self.gine1 = GINEConv(
            nn = torch.nn.Sequential(
                torch.nn.Linear(num_node_features, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU()
            ),
            train_eps=True,
            edge_dim=1
        )
        self.gine2 = GINEConv(
            nn = torch.nn.Sequential(
                torch.nn.Linear(32, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU()
            ),
            train_eps=True,
            edge_dim=1
        )
        self.gine3 = GINEConv(
            nn = torch.nn.Sequential(
                torch.nn.Linear(32, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU()
            ),
            train_eps=True,
            edge_dim=1
        )
        self.state_fc = torch.nn.Linear(num_state_features, 16)
        self.actor_head1 = torch.nn.Linear(32 + 16, 32)
        self.actor_head2 = torch.nn.Linear(32, num_actions)
        self.value_head1 = torch.nn.Linear(2*32 + 16, 32)
        self.value_head2 = torch.nn.Linear(32, 1)

    def forward(self, states, graph_data):
        if isinstance(graph_data, list):
            mini_batch = Batch.from_data_list(graph_data)
            graph_x, edge_index, edge_attr, batch = mini_batch.x, mini_batch.edge_index, mini_batch.edge_attr, mini_batch.batch
        elif isinstance(graph_data, Data):
            graph_x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
        else:
            raise TypeError("unsupported datatype")
        graph_x = self.gine1(graph_x, edge_index, edge_attr)
        graph_x = self.gine2(graph_x, edge_index, edge_attr)
        graph_x = self.gine3(graph_x, edge_index, edge_attr)

        graph_x_mean = global_mean_pool(graph_x, batch=batch)
        graph_x_max = global_max_pool(graph_x, batch=batch)
        state_x = self.state_fc(states)
        state_x = torch.nn.functional.relu(state_x)
        self.intermediate_state = torch.cat((state_x.unsqueeze(0), graph_x_mean, graph_x_max), dim=1)

        graph_x = torch.cat((graph_x, state_x.repeat(graph_x.shape[0], 1)), dim=1)
        graph_x = self.actor_head1(graph_x)
        graph_x = torch.nn.functional.relu(graph_x)
        graph_x = self.actor_head2(graph_x)
        if self.config["global_softmax"]:
            graph_x = torch.nn.functional.softmax(graph_x.flatten(), dim=0).reshape(graph_x.shape) # make softmax over all nodes and actions in order to not sample the node uniform at random
        else:
            graph_x = torch.nn.functional.softmax(graph_x, dim=-1)

        return graph_x

    def value_function(self):
        assert hasattr(self, "intermediate_state"), "can not call value_function before forward"
        val = self.value_head1(self.intermediate_state)
        val = torch.nn.functional.relu(val)
        return self.value_head2(val)


class GCN(TorchModelV2, torch.nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space, 
        action_space: gym.spaces.Space, 
        num_outputs: int, 
        model_config: ModelConfigDict, 
        name: str,
        lstm_state_size=32,
        **custom_model_kwargs
    ):
        torch.nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        self.use_graph = custom_model_kwargs["use_graph"]
        self.use_previous_action = custom_model_kwargs["use_previous_action"]
        self.use_state = custom_model_kwargs["use_state"]
        assert (self.use_graph or self.use_previous_action or self.use_state), "Some features are required!"
        self.num_outputs = num_outputs
        self.lstm_state_size = lstm_state_size
        original_obs_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        self.states_size = original_obs_space.spaces["states"].shape[0]
        # self.num_node_features = original_obs_space.spaces["node_features"].shape[1]
        # self.num_edge_features = original_obs_space.spaces["edge_attr"].shape[1]
        # create the GINE GCNs
        if self.use_graph:
            GINE_OUT = 8
            self.gine1 = GINEConv(
                nn = torch.nn.Sequential(
                    torch.nn.Linear(original_obs_space["node_features"].shape[1], 16),
                    torch.nn.BatchNorm1d(16),
                    torch.nn.ReLU()
                ),
                train_eps=True,
                edge_dim=1
            )
            self.gine2 = GINEConv(
                nn = torch.nn.Sequential(
                    torch.nn.Linear(16, 16),
                    torch.nn.BatchNorm1d(16),
                    torch.nn.ReLU()
                ),
                train_eps=True,
                edge_dim=1
            )
            self.gine3 = GINEConv(
                nn = torch.nn.Sequential(
                    torch.nn.Linear(16, GINE_OUT),
                    torch.nn.ReLU()
                ),
                train_eps=True,
                edge_dim=1
            )
        else:
            GINE_OUT = 0

        # create the LSTM layer
        if self.use_previous_action:
            ACTION_OUT = 8
            self.lstm = torch.nn.LSTM(self.num_outputs + 1, self.lstm_state_size, batch_first=True)
            self.action_fc = torch.nn.Linear(self.lstm_state_size, ACTION_OUT)
        else:
            ACTION_OUT = 0

        if self.use_state:
            # create the layers for the state processing
            STATE_OUT = 12
            self.state_fc = torch.nn.Linear(self.states_size, STATE_OUT)
        else:
            STATE_OUT = 0

        # create the layers for state postprocessing
        self.state_postprocessing_fc = torch.nn.Linear(GINE_OUT + GINE_OUT + STATE_OUT + ACTION_OUT, 32)

        # create the layers for the policy network
        self.policy_fc1 = torch.nn.Linear(32, 32)
        self.policy_fc2 = torch.nn.Linear(32, self.num_outputs)

        # create the layers for the value network
        self.value_fc1 = torch.nn.Linear(32, 32)
        self.value_fc2 = torch.nn.Linear(32, 1)

    def forward(self, input_dict, state, seq_lens):
        # handle reshaping of all the data according to seq_lens
        new_batch_size = seq_lens.shape[0]
        batch_size = input_dict.count
        time_size = batch_size // new_batch_size
        assert new_batch_size * time_size == batch_size, f"seq_lens {seq_lens} and batch size {batch_size} don't add up, input probably has to be zero padded"
        batch_major_shape = (new_batch_size, time_size)

        # handle the graph data
        if self.use_graph:
            x = input_dict["obs"]["node_features"]
            edge_index = input_dict["obs"]["edge_index"]
            edge_attr = input_dict["obs"]["edge_attr"]
            node_data_size =  input_dict["obs"]["node_data_size"].int().squeeze(1)
            edge_data_size = input_dict["obs"]["edge_data_size"].int().squeeze(1)
            data_samples = []
            for x_sample, edge_index_sample, edge_attr_sample, node_data_size_sample, edge_data_size_sample in zip(x, edge_index, edge_attr, node_data_size, edge_data_size):
                # handle stupid initialization calls
                if node_data_size_sample == 0 and edge_data_size_sample == 0:
                    node_data_size_sample = 1
                    edge_data_size_sample = 1
                data_samples.append(Data(x=x_sample[:node_data_size_sample], edge_index=edge_index_sample[:, :edge_data_size_sample], edge_attr=edge_attr_sample[:edge_data_size_sample]))
            batch = Batch.from_data_list(data_samples)

            graph_x, edge_index, edge_attr = batch.x, batch.edge_index.long(), batch.edge_attr
            graph_x = self.gine1(graph_x, edge_index, edge_attr)
            graph_x = self.gine2(graph_x, edge_index, edge_attr)
            graph_x = self.gine3(graph_x, edge_index, edge_attr)
            graph_x_mean = global_mean_pool(graph_x, batch=batch.batch)
            graph_x_max = global_max_pool(graph_x, batch=batch.batch)

        # handle the global states data
        if self.use_state:
            states_x = input_dict["obs"]["states"]
            states_x = self.state_fc(states_x)
            states_x = F.relu(states_x)

        # handle the action history
        if self.use_previous_action:
            previous_actions = input_dict["obs"]["previous_action"]
            previous_actions_shape = previous_actions.shape[1:]
            previous_actions = previous_actions.view(batch_major_shape + previous_actions_shape)
            prev_action_one_hot = torch.nn.functional.one_hot(previous_actions.long(), num_classes=self.num_outputs + 1).float()
            prev_action_one_hot = prev_action_one_hot.squeeze(2)
            h = torch.swapaxes(state[0], 0, 1)
            c = torch.swapaxes(state[1], 0, 1)
            action_x, [h, c] = self.lstm(prev_action_one_hot, [h, c])
            action_x = action_x.reshape((batch_size, self.lstm_state_size))
            action_x = self.action_fc(action_x)
            action_x = F.relu(action_x)

        # get the action
        preprocessed_signals = []
        if self.use_state:
            preprocessed_signals.append(states_x)
        if self.use_graph:
            preprocessed_signals.extend([graph_x_max, graph_x_mean])
        if self.use_previous_action:
            preprocessed_signals.append(action_x)
        self.intermediate_x = self.state_postprocessing_fc(torch.cat(preprocessed_signals, dim=1))

        policy_x = self.policy_fc1(self.intermediate_x)
        policy_x = self.policy_fc2(policy_x)
        
        if self.use_previous_action:
            return (policy_x, [h, c])
        else:
            return (policy_x, state)

    def value_function(self) -> TensorType:
        assert self.intermediate_x is not None, "must call forward() first"
        value = self.value_fc1(self.intermediate_x)
        return self.value_fc2(value).view(-1)

    def get_initial_state(self):
        h = [
            torch.zeros(1, self.lstm_state_size),
            torch.zeros(1, self.lstm_state_size),
        ]
        return h