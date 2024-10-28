from abc import ABC
import torch
import torch.nn as nn
from torchprofile import profile_macs

class CentralizedPPOCritic(nn.Module, ABC):

    DEFAULTS = {} 
    def __init__(self, params):
        super(CentralizedPPOCritic, self).__init__()

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(CentralizedPPOCritic.DEFAULTS, **params_dict)

        if self.use_cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_shape = self.num_agents*4
        self.output_shape = 1

        self.fc1 = nn.Linear(self.input_shape, self.critic_hidden)
        self.rnn = nn.GRUCell(self.critic_hidden, self.critic_hidden)
        self.fc2 = nn.Linear(self.critic_hidden, self.output_shape)

    def init_hidden(self, hidden_layer1 = None):
    
        if hidden_layer1 == None:
            self.hidden_layer1 = torch.zeros((self.batch_size, self.critic_hidden)).to(self.device)
        else:
            self.hidden_layer1 = hidden_layer1
            
    def forward(self, state, hidden_state):
        state = state.view(-1, self.num_agents*4)
        fc1_out = torch.relu(self.fc1(state))
        h_in = hidden_state.reshape(-1, self.critic_hidden)
        rnn_out = self.rnn(fc1_out, h_in)
        fc2_out = self.fc2(rnn_out) # torch.relu(self.fc2(rnn_out))
        fc2_out = fc2_out.view(-1, self.output_shape)
        return fc2_out, rnn_out

    def print_MACs_FLOPs(self):
        num_macs = profile_macs(self, (torch.zeros(1, self.num_agents*4), torch.zeros(1, self.critic_hidden))) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)

class CentralizedPPOActor(nn.Module, ABC):

    DEFAULTS = {} 
    def __init__(self, params):
        super(CentralizedPPOActor, self).__init__()

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(CentralizedPPOActor.DEFAULTS, **params_dict)

        # MODIFY TO INSERT THE NUMBER OF NEIGHBORS AGENTS
        self.input_size = 4*self.LSTM_hidden*2 # 4 input_feat * 2 (hidden state + final cell state)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_agents),
            nn.Sigmoid()
        )

    def forward(self, hidden):
        result = self.fc(hidden).squeeze()

        # Clip the probabilities to avoid log(0) or log(1) issues
        result = torch.clamp(result, self.actor_epsilon_clip, 1 - self.actor_epsilon_clip)
        return result
    
    def print_MACs_FLOPs(self):
        num_macs = profile_macs(self, torch.zeros(1, self.input_size)) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)