import os
import sys
import torch.nn as nn
import torch
from torchprofile import profile_macs

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(cwd))

from Classes.utils.utils import return_tensor

class LSTM(nn.Module):

    DEFAULTS = {} 
    def __init__(self, params):
        super(LSTM, self).__init__()

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(LSTM.DEFAULTS, **params_dict)

        if self.use_cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_size = 4 + self.num_agents + self.num_features + self.num_agents # GNSS + A2A + A2F + Action (num_agents)
        self.output_size = 4

        self.LSTM_bidirectional_dimension = 2 if self.LSTM_bidirectional else 1

        # LSTM LAYER
        self.lstm_layer1 = nn.LSTM(input_size=self.input_size,hidden_size=self.LSTM_hidden, num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer1_act = nn.ReLU(inplace=False)     
        self.lstm_layer2 = nn.LSTM(input_size=self.LSTM_hidden*self.LSTM_bidirectional_dimension,hidden_size=self.LSTM_hidden, num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer2_act = nn.ReLU(inplace=False)     
        self.lstm_maxout1 = Maxout(self.LSTM_hidden*self.LSTM_bidirectional_dimension, 128, 2)
        self.lstm_linear1 = nn.Linear(128, 64)
        self.lstm_linear1_act = nn.ReLU(inplace=False)     
        self.lstm_linear2 = nn.Linear(64, 32)
        self.lstm_linear3 = nn.Linear(32, self.output_size)


    def init_hidden(self, hidden_layer1 = None):

        # Layer 1
        if hidden_layer1 == None:
            self.hidden_layer1 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size, self.LSTM_hidden, requires_grad=True).to(self.device),
                    torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size, self.LSTM_hidden, requires_grad=True).to(self.device))
        else:
            self.hidden_layer1 = hidden_layer1

        # Layer 2
        # Initialize hidden layer 2
        self.hidden_layer2 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size, self.LSTM_hidden, requires_grad=True).to(self.device),
                            torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size, self.LSTM_hidden, requires_grad=True).to(self.device))
    
    def forward(self, x, hidden_layer1, use_cuda = None):
        
        batch_shape = x.shape[0]
        if use_cuda == None:
            use_cuda = self.use_cuda

        x = return_tensor(x, use_cuda = use_cuda)

        ###### INFO LSTM ######
        # Hidden state:
        # always in the shape ((2/1 (mono-bi directional) * num_layers, B, H_out=H_hidden),
        #                      (2/1 (mono-bi directional) * num_layers, B, H_out=H_hidden)) 
        # hidden = (torch.randn(1, self.num_agents, 4),
        #         torch.randn(1, self.num_agents, 4))

        # INPUT: (B: batch_size, L: length seq, Hin: num input features)
        # B = self.num_agents
        # L = 1 (manually step through the sequence one element at a time.)
        # Hin = 6 (x, y, vx, vy)

        # OUTPUT: (B: batch_size, L: length seq, H_out=H_hidden)
        ########################


        # LSTM LAYER
        self.hidden_layer2 = (self.hidden_layer2[0].detach(), self.hidden_layer2[1].detach())

        x_lstm, hidden_layer1 = self.lstm_layer1(x.view(self.num_agents*self.batch_size, 1, self.input_size), hidden_layer1)
        x_lstm = self.lstm_layer1_act(x_lstm)
        x_lstm, self.hidden_layer2 = self.lstm_layer2(x_lstm, self.hidden_layer2)
        x_lstm = self.lstm_layer2_act(x_lstm)
        x_lstm = self.lstm_maxout1(x_lstm)
        x_lstm = self.lstm_linear1(x_lstm)
        x_lstm = self.lstm_linear1_act(x_lstm)
        x_lstm = self.lstm_linear2(x_lstm)
        x_lstm = self.lstm_linear3(x_lstm)

        return x_lstm, hidden_layer1


    def print_MACs_FLOPs(self):

        num_macs = profile_macs(self, (torch.zeros(self.num_agents*self.batch_size, 1, self.input_size), self.hidden_layer1)) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)

class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.pool_size = pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, x):
        shape = list(x.shape)
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        x = self.lin(x)
        x = x.view(*shape)
        x = x.max(-1)[0]
        return x