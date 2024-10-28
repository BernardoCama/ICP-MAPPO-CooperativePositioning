import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0], 'Classes')
EXPERIMENTS_DIR = os.path.join(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0], 'Exp')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(cwd))

from Classes.MARL.models.LSTM import LSTM
from Classes.MARL.models.ppo_net import CentralizedPPOActor, CentralizedPPOCritic

from Classes.utils.utils import return_tensor, return_numpy, normalize_numpy

class ICPMAPPO(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(ICPMAPPO.DEFAULTS, **params_dict)

        self.ppo_lstm = LSTM(self.params)
        self.ppo_actor = CentralizedPPOActor(self.params)
        self.ppo_critic = CentralizedPPOCritic(self.params)

        self.optimizer_lstm = torch.optim.Adam(params=self.ppo_lstm.parameters(),
                                                lr=self.lr_lstm)
        self.optimizer_actor = torch.optim.Adam(params=self.ppo_actor.parameters(),
                                                lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(params=self.ppo_critic.parameters(),
                                                 lr=self.lr_critic)

        self.result_train_path = os.path.join(self.output_results_dir, 'output_ICPMAPPO_train_results.npy')
        self.result_test_path = os.path.join(self.output_results_dir, 'output_ICPMAPPO_test_results.npy')
        self.ppo_lstm_path = os.path.join(self.saved_models_dir, "ppo_lstm.pth")
        self.ppo_actor_path = os.path.join(self.saved_models_dir, "ppo_actor.pth")
        self.ppo_critic_path = os.path.join(self.saved_models_dir, "ppo_critic.pth")

        self.set_cuda_device()

        self.set_model_cuda(self.ppo_lstm)
        self.set_model_cuda(self.ppo_actor)
        self.set_model_cuda(self.ppo_critic)
        
        self.cov_var = torch.full(size=(2,), fill_value=0.01)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

    def set_cuda_device(self, use_cuda = None):
        if use_cuda == None:
            use_cuda = self.use_cuda
        
        if use_cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    # Set model to GPU
    def set_model_cuda(self, model = None, use_cuda = None):
        
        if use_cuda == None:
            use_cuda = self.use_cuda

        if model == None:
            if use_cuda:
                self.ppo_lstm.to("cuda:0", non_blocking=True)
                self.ppo_actor.to("cuda:0", non_blocking=True)
                self.ppo_critic.to("cuda:0", non_blocking=True)
            else:
                self.ppo_lstm.cpu()
                self.ppo_actor.cpu()
                self.ppo_critic.cpu()
        else:
            if use_cuda:
                model.to("cuda:0", non_blocking=True)
            else:
                model.cpu() 


    def learn_actor_critic(self, batch_data: dict, episode_num: int = 0):

        # timesteps x num_agents*batch_size x  (4 + self.num_agents + self.num_features + self.num_agents) # GNSS + A2A + A2F 
        obs = batch_data['obs'].to(self.device).detach()
        # timesteps x num_agents*batch_size x 4 (input_features)
        state = batch_data['state'].to(self.device)
        state_hat = batch_data['state_hat'].to(self.device)
        # timesteps x num_agents*batch_size x num_agents 
        actions = batch_data['actions'].to(self.device)
        obs_next = batch_data['obs_next'].to(self.device).detach()
        state_next = batch_data['state_next'].to(self.device)
        state_hat_next = batch_data['state_hat_next'].to(self.device)
        # timesteps x num_agents*batch_size x num_agents 
        log_probs = batch_data['log_probs'].to(self.device)
        rewards = batch_data['rewards']
        # timesteps x 2 (hidden state + final cell state) x 4 x num_agents*batch_size x hidden_dimension
        hidden_lstm = batch_data['hidden_lstm']

        per_episode_len = sum(batch_data['per_episode_len'])
        # obs = obs.reshape(batch_size, -1)

        # Compute rawards-to-go
        discount_reward = self.get_discount_reward(rewards).to(self.device)
        
        self.ppo_critic.init_hidden()
        hidden_critic = self.ppo_critic.hidden_layer1
        with torch.no_grad():

            advantage_function = []
            for i in range(per_episode_len):

                state_value, hidden_critic = self.ppo_critic(state[i,:,:], hidden_critic.detach())
                advantage_function_single = discount_reward[i, :].reshape(-1, 1) - state_value.reshape(-1, 1)
                # Advantage_function
                if not torch.isnan(advantage_function_single.std()):
                    advantage_function.append(((advantage_function_single - advantage_function_single.mean()) / (
                            advantage_function_single.std() + 1e-10)).unsqueeze(dim=-1))
                else:
                    advantage_function.append(((advantage_function_single)).unsqueeze(dim=-1))             
                
            # timesteps 
            advantage_function = torch.stack(advantage_function).view(per_episode_len, self.batch_size)
            

        self.ppo_critic.init_hidden()
        hidden_critic = self.ppo_critic.hidden_layer1

        actor_losses = []
        critic_losses = []
        state_values = []
        ratios_mean = []
        advantage_function_mean = []

        for i in range(per_episode_len):
                
            # curr_log_probs: num_agents*batch_size (batch) x num_agents (action dimension) x 2 (binary action space), entropy: num_agents (batch) x num_agents (action dimension)
            _, curr_log_probs, curr_entropy = self.choose_actions(hidden_lstm[i], train = 1)

            # loss
            # num_agents (batch) x num_agents (action dimension)
            # Consider the full distribution over actions 
            ratios = torch.exp(curr_log_probs - log_probs[i, :, :, :].squeeze()).reshape(self.batch_size, self.num_agents, self.num_agents, 2)
            surr1 = ratios * advantage_function[i, :].reshape(-1, 1, 1, 1)
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip,
                                1 + self.epsilon_clip) * advantage_function[i,:].reshape(-1, 1, 1, 1)
            actor_loss = - (torch.min(surr1, surr2).mean() + self.alpha_entropy * curr_entropy.mean())

            # actor_loss:
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_actor.parameters(), self.actor_max_grad_norm)
            self.optimizer_actor.step()

            # critic_loss: td_error
            curr_state_value, hidden_critic = self.ppo_critic(state[i,:,:], hidden_critic.detach())
            critic_loss = nn.MSELoss()(curr_state_value.view(self.batch_size), discount_reward[i, :].view(self.batch_size))
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # Store results
            actor_losses.append(actor_loss.cpu().detach().numpy())
            critic_losses.append(critic_loss.cpu().detach().numpy())
            state_values.append(curr_state_value.cpu().detach().numpy())
            ratios_mean.append(ratios.mean().cpu().detach().numpy())
            advantage_function_mean.append(advantage_function.mean().cpu().detach().numpy())
        
        return actor_losses, critic_losses, state_values, ratios_mean, advantage_function_mean



    def learn_lstm(self, env, state_hat_next, state_next, train = 1):
        
        state_next = return_tensor(state_next, use_cuda = self.use_cuda)
        loss = torch.mean((state_hat_next-state_next)**2)

        if train:
            self.optimizer_lstm.zero_grad()
            loss.backward()
            self.optimizer_lstm.step()

        state_next_original_space = deepcopy(state_next.detach())
        state_hat_next_original_space = deepcopy(state_hat_next.detach())

        state_next_original_space[:,0:2] = normalize_numpy(state_next[:,0:2], np.array([env.limit_pos1[0], env.limit_pos2[0]]), np.array([env.limit_pos1[1], env.limit_pos2[1]]), normalize = 0, type_='minmax', already_centered = 0)
        state_next_original_space[:,2:4] = normalize_numpy(state_next[:,2:4], np.array([env.limit_vel1[0], env.limit_vel2[0]]), np.array([env.limit_vel1[1], env.limit_vel2[1]]), normalize = 0, type_='minmax', already_centered = 0)

        state_hat_next_original_space[:,0:2] = normalize_numpy(state_hat_next[:,0:2].detach(), np.array([env.limit_pos1[0], env.limit_pos2[0]]), np.array([env.limit_pos1[1], env.limit_pos2[1]]), normalize = 0, type_='minmax', already_centered = 0)
        state_hat_next_original_space[:,2:4] = normalize_numpy(state_hat_next[:,2:4].detach(), np.array([env.limit_vel1[0], env.limit_vel2[0]]), np.array([env.limit_vel1[1], env.limit_vel2[1]]), normalize = 0, type_='minmax', already_centered = 0)

        # Sum Square Errors
        SE_pos = torch.sum((state_next_original_space[:,0:2]-state_hat_next_original_space[:,0:2])**2, 1) # (num_agents*batch_size)
        RMSE_pos = torch.sqrt(torch.mean(SE_pos)) # (1)
        SE_vel = torch.sum((state_next_original_space[:,2:4]-state_hat_next_original_space[:,2:4])**2, 1) # (num_agents*batch_size)
        RMSE_vel = torch.sqrt(torch.mean(SE_vel)) # (1)

        return SE_pos.detach(), RMSE_pos.detach(), SE_vel.detach(), RMSE_vel.detach(), loss.detach()
    
    def choose_actions(self, hidden_lstm: dict, train = 0, connectivity_matrix_A2A = None):

        actions = []

        # From  ((4 (num_inputs), num_agents*batch_size (batch), lstm_hidden), (4 (num_inputs), num_agents*batch_size (batch), lstm_hidden))
        # to    (num_agents*batch_size, 4*lstm_hidden*2)
        hidden_lstm = torch.concat((torch.swapaxes(hidden_lstm[0], 0, 1).reshape(self.num_agents*self.batch_size, -1), 
                                   torch.swapaxes(hidden_lstm[1], 0, 1).reshape(self.num_agents*self.batch_size, -1)), 1).to(self.device)

        if train:
            action_prop = self.ppo_actor(hidden_lstm)
        else:
            with torch.no_grad():
                self.ppo_actor.eval()
                action_prop = self.ppo_actor(hidden_lstm)

        if connectivity_matrix_A2A is None:
            actions = np.zeros_like(action_prop.cpu().detach())
            for i, agent_name in enumerate(range(self.num_agents*self.batch_size)):
                for j, agent_name in enumerate(range(self.num_agents)):
        
                    action = 1 if random.random() < action_prop[i,j] else 0  # Randomly choose action based on probability
                    actions[i,j] = int(action)
        
        # Use the connectivity_matrix_A2A to correct unfeasibility of actions
        else:
            actions = np.zeros_like(action_prop.cpu().detach())
            for i, agent_name in enumerate(range(self.num_agents*self.batch_size)):
                for j, agent_name in enumerate(range(self.num_agents)):
                    if connectivity_matrix_A2A[i, j]  == 1:
                        action = 1 if random.random() < action_prop[i,j] else 0  # Randomly choose action based on probability
                        actions[i,j] = int(action)   
                    else:
                        action_prop[i,j] = 1e-9
                        action = 1 if random.random() < action_prop[i,j] else 0  # Randomly choose action based on probability
                        actions[i,j] = int(action)   

        log_prob = torch.log(action_prop)
        log_prob_1_minus = torch.log(1-action_prop)
        entropy = -(action_prop * log_prob + (1 - action_prop) * log_prob_1_minus)

        return actions,  torch.stack((log_prob, log_prob_1_minus),2), entropy
    
    def get_state_estimate(self, actions, hidden_lstm, obs_next, connectivity_matrix_A2A, train = 1):

        hidden_lstm_new = (hidden_lstm[0].clone().detach().requires_grad_(False),
                           hidden_lstm[1].clone().detach().requires_grad_(False))

        # Add only hidden of connected agents and modify action with -1 in case of no-connectivity
        for i in range(self.num_agents*self.batch_size): 
            num_selected_agents = 1
            for j in range(self.num_agents):
                # The two agents can exchange hidden features
                if connectivity_matrix_A2A[i,j] == 1:
                    if actions[i][j] == 1 and i !=j:
                        hidden_lstm_new[0][:, i, :] = hidden_lstm_new[0][:, i, :] + hidden_lstm[0][:, j, :]
                        hidden_lstm_new[1][:, i, :] = hidden_lstm_new[1][:, i, :] + hidden_lstm[1][:, j, :]
                        num_selected_agents += 1
                # There is no connection between the two agents
                else:
                    actions[i][j] = -1      

            if num_selected_agents:
                hidden_lstm_new[0][:, i, :] = hidden_lstm_new[0][:, i, :]/num_selected_agents
                hidden_lstm_new[1][:, i, :] = hidden_lstm_new[1][:, i, :]/num_selected_agents

        # batch (num_agents*batch_size) x input_feature (4 + self.num_agents + self.num_features + self.num_agents)
        input_lstm = np.concatenate((obs_next, actions), 1)

        if train:
            state_hat_next, hidden_lstm_next = self.ppo_lstm(input_lstm, hidden_lstm_new)
        else:
            with torch.no_grad():
                self.ppo_lstm.eval()
                state_hat_next, hidden_lstm_next = self.ppo_lstm(input_lstm, hidden_lstm_new)

        return state_hat_next, hidden_lstm_next
    

    def get_state_estimate_single_agent(self, actions, hidden_lstm, obs_next, connectivity_matrix_A2A, train = 1):

        # Keep original hidden features
        hidden_lstm_new = (hidden_lstm[0].clone().detach().requires_grad_(False),
                           hidden_lstm[1].clone().detach().requires_grad_(False))

        # Do not communicate with anyone
        actions = np.zeros_like(actions) - 1

        # batch (num_agents*batch_size) x input_feature (4 + self.num_agents + self.num_features + self.num_agents)
        input_lstm = np.concatenate((obs_next, actions), 1)

        if train:
            state_hat_next, hidden_lstm_next = self.ppo_lstm(input_lstm, hidden_lstm_new)
        else:
            with torch.no_grad():
                state_hat_next, hidden_lstm_next = self.ppo_lstm(input_lstm, hidden_lstm_new)

        return state_hat_next, hidden_lstm_next
    

    def get_reward(self, env, state, state_hat, state_next, state_hat_next):

        state = return_numpy(state).reshape(self.batch_size, self.num_agents, 4)
        state_hat = return_numpy(state_hat).reshape(self.batch_size, self.num_agents, 4)
        state_next = return_numpy(state_next).reshape(self.batch_size, self.num_agents, 4)
        state_hat_next = return_numpy(state_hat_next).reshape(self.batch_size, self.num_agents, 4)

        beta_reward = np.mean(normalize_numpy(self.beta_reward, np.array([env.limit_pos1[0], env.limit_pos2[0]]), np.array([env.limit_pos1[1], env.limit_pos2[1]]), normalize = 1, type_='minmax', already_centered = 1))

        value = np.mean(np.mean((state_hat-state)**2 - (state_hat_next-state_next)**2, 1), 1)
        
        value[value <= -beta_reward] = -1
        value[value > beta_reward] = 1
        value[(value <= beta_reward) & (value > - beta_reward)] = 2
        return value


    def get_discount_reward(self, batch_reward: list) -> Tensor:

        batch_reward = batch_reward.tolist()
        discount_rewards = []
        for i, reward in enumerate(batch_reward):
            discount_rewards.append([])
            discounted_reward = 0
            for one_reward in reversed(reward):
                discounted_reward = one_reward + discounted_reward * self.gamma_discount
                discount_rewards[i].insert(0, discounted_reward)
        return torch.Tensor(discount_rewards)
    

    def save_model_and_train_result(self, step = None, train_results = None, saved_models_dir = None):
        if train_results is not None:
            np.save(self.result_train_path, train_results, allow_pickle = True)
        self.save_model(saved_models_dir = saved_models_dir)

    def load_model_and_train_result(self, saved_models_dir = None):
        self.load_model(saved_models_dir = saved_models_dir)  
        try: 
            return np.load(self.result_train_path, allow_pickle = True).tolist()
        except:
            return None
    
    def save_test_result(self, test_results):
        np.save(self.result_test_path, test_results, allow_pickle = True)

    def load_test_result(self):
        return np.load(self.result_test_path, allow_pickle = True).tolist()

    def save_model(self, saved_models_dir = None):
        if saved_models_dir == None:
            torch.save(self.ppo_lstm.state_dict(), self.ppo_lstm_path)
            torch.save(self.ppo_actor.state_dict(), self.ppo_actor_path)
            torch.save(self.ppo_critic.state_dict(), self.ppo_critic_path)
        else:
            ppo_lstm_path = os.path.join(saved_models_dir, "ppo_lstm.pth")
            ppo_actor_path = os.path.join(saved_models_dir, "ppo_actor.pth")
            ppo_critic_path = os.path.join(saved_models_dir, "ppo_critic.pth")
            torch.save(self.ppo_lstm.state_dict(), ppo_lstm_path)
            torch.save(self.ppo_actor.state_dict(), ppo_actor_path)
            torch.save(self.ppo_critic.state_dict(), ppo_critic_path)

    def load_model(self, saved_models_dir = None):
        if saved_models_dir == None:
            self.ppo_lstm.load_state_dict(torch.load(self.ppo_lstm_path, map_location=torch.device(self.device)))
            self.ppo_actor.load_state_dict(torch.load(self.ppo_actor_path, map_location=torch.device(self.device)))
            self.ppo_critic.load_state_dict(torch.load(self.ppo_critic_path, map_location=torch.device(self.device)))
        else:
            ppo_lstm_path = os.path.join(saved_models_dir, "ppo_lstm.pth")
            ppo_actor_path = os.path.join(saved_models_dir, "ppo_actor.pth")
            ppo_critic_path = os.path.join(saved_models_dir, "ppo_critic.pth")
            self.ppo_lstm.load_state_dict(torch.load(ppo_lstm_path, map_location=torch.device(self.device)))
            self.ppo_actor.load_state_dict(torch.load(ppo_actor_path, map_location=torch.device(self.device)))
            self.ppo_critic.load_state_dict(torch.load(ppo_critic_path, map_location=torch.device(self.device)))

    def del_model(self, saved_models_dir = None):
        if saved_models_dir == None:
            file_list = os.listdir(self.saved_models_dir)
            for file in file_list:
                if file.split('.')[-1] == 'pth':
                    os.remove(os.path.join(self.saved_models_dir, file))
        else:
            file_list = os.listdir(saved_models_dir)
            for file in file_list:
                if file.split('.')[-1] == 'pth':
                    os.remove(os.path.join(saved_models_dir, file))

