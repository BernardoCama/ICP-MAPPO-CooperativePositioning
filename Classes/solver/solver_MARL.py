import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")
from tqdm import tqdm
from copy import deepcopy
import torch

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'Classes')
EXPERIMENTS_DIR = os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'Exp')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(cwd))
from Classes.utils.utils import  normalize_numpy, mkdir
from Classes.MARL.common.reply_buffer import CommBatchEpisodeMemory
from Classes.MARL.ICPMAPPO_algorithm.ICPMAPPO import ICPMAPPO

class Solver_MARL(object):
    
    DEFAULTS = {}   
    def __init__(self, params, env):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(Solver_MARL.DEFAULTS, **params_dict)

        self.env = env

        self.batch_episode_memory = CommBatchEpisodeMemory(continuous_actions=False,
                                                            n_actions=2,
                                                            n_agents=self.num_agents)
    
        self.agents = ICPMAPPO(self.params)



    def train(self, train_results = {}):

        # Batch: num_agents (* number of environments )
        if len(list(train_results.keys())) != 0:
            start_epoch = max(list(train_results.keys())) + 1
        else:
            start_epoch = 0

        for step in tqdm(range(0+start_epoch, self.num_steps+start_epoch)):

            train_results[step] = {}
            
            train_loader, train_numpy_dataset, train_raw_dataset = self.env.return_dataset_tracking(train = 1, normalize = 1, shuffle = 0, bool_save_dataset = 0, bool_load_dataset = 0)
            train_dict_dataset = {dataset_output_name:train_numpy_dataset[index_name] for index_name,(dataset_output_name) in enumerate(self.env.dataset_output_names)}

            # reshape according to the sequence length
            train_dict_dataset['t_A'] = train_dict_dataset['t_A'].reshape(self.batch_size, self.L, self.num_agents, 4)
            train_dict_dataset['x_GNSS'] = train_dict_dataset['x_GNSS'].reshape(self.batch_size, self.L, self.num_agents, 4)
            train_dict_dataset['x_A2A'] = train_dict_dataset['x_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
            train_dict_dataset['x_A2F'] = train_dict_dataset['x_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)
            train_dict_dataset['mutual_distances_A2A'] = train_dict_dataset['mutual_distances_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
            train_dict_dataset['mutual_distances_A2F'] = train_dict_dataset['mutual_distances_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)
            train_dict_dataset['connectivity_matrix_A2A'] = train_dict_dataset['connectivity_matrix_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
            train_dict_dataset['connectivity_matrix_A2F'] = train_dict_dataset['connectivity_matrix_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)

            L = train_dict_dataset['t_A'].shape[1]

            state = train_dict_dataset['t_A'][:, 0].reshape(-1, 4)
            state_hat = torch.Tensor(state)
            # num_agents x (1 + num_agents + num_features)
            obs = np.concatenate((train_dict_dataset['x_GNSS'][:, 0].reshape(-1, 4), 
                                 train_dict_dataset['x_A2A'][:, 0].reshape(-1, self.num_agents),
                                 train_dict_dataset['x_A2F'][:, 0].reshape(-1, self.num_features)), 1)
            
            # Subsitute nan values in x_A2A and x_A2F with -1:
            train_dict_dataset['x_A2A'][np.where(np.isnan(train_dict_dataset['x_A2A']))] = -1
            train_dict_dataset['x_A2F'][np.where(np.isnan(train_dict_dataset['x_A2F']))] = -1
            
            self.agents.ppo_lstm.init_hidden()
            hidden_lstm = self.agents.ppo_lstm.hidden_layer1

            SEs_pos = []
            RMSEs_pos = []
            SEs_vel = []
            RMSEs_vel = []

            total_rewards = []
            lstm_losses = []

            for t in range(1, L):

                actions, log_probs, entropy = self.agents.choose_actions(hidden_lstm, train = 0)
                log_probs = log_probs.detach()

                connectivity_matrix_A2A = train_dict_dataset['connectivity_matrix_A2A'][:, t].reshape(-1, self.num_agents)

                obs_next = np.concatenate((train_dict_dataset['x_GNSS'][:, t].reshape(-1, 4), 
                                    train_dict_dataset['x_A2A'][:, t].reshape(-1, self.num_agents),
                                    train_dict_dataset['x_A2F'][:, t].reshape(-1, self.num_features)), 1)
                
                state_next = train_dict_dataset['t_A'][:, t].reshape(-1, 4)

                state_hat_next, hidden_lstm_next = self.agents.get_state_estimate(actions, hidden_lstm, obs_next, connectivity_matrix_A2A, train = 1)
                state_hat_next = state_hat_next.view(self.num_agents*self.batch_size, 4)

                # Update world model -> train LSTM
                # num_agents x 1, 1, num_agents x 1, 1
                SE_pos, RMSE_pos, SE_vel, RMSE_vel, lstm_loss = self.agents.learn_lstm(self.env, state_hat_next, state_next, train = 1)

                rewards = self.agents.get_reward(self.env, state, state_hat, state_next, state_hat_next)

                # Store transition
                self.batch_episode_memory.store_one_episode(one_state=state, one_state_hat=state_hat, one_obs=obs, action=actions, reward=rewards, 
                                                            one_state_next=state_next, one_obs_next=obs_next, one_state_hat_next=state_hat_next,
                                                            log_probs = log_probs, hidden_lstm=hidden_lstm)
                # Next timestep
                total_rewards.append(rewards)
                state = state_next
                state_hat = state_hat_next
                obs = obs_next
                hidden_lstm = (hidden_lstm_next[0].detach(), hidden_lstm_next[1].detach())

                # Store results
                SEs_pos.append(SE_pos.cpu().numpy())
                RMSEs_pos.append(RMSE_pos.cpu().numpy())
                SEs_vel.append(SE_vel.cpu().numpy())
                RMSEs_vel.append(RMSE_vel.cpu().numpy())

                lstm_losses.append(lstm_loss.cpu().numpy())

            self.batch_episode_memory.set_per_episode_len(t)

            batch_data = self.batch_episode_memory.get_batch_data()
            actor_losses, critic_losses, state_values, ratios_mean, advantage_function_mean = self.agents.learn_actor_critic(batch_data)
            self.batch_episode_memory.clear_memories()

            # mean_reward = self.evaluate()
            mean_reward = np.mean(total_rewards)
            mean_RMSE_pos = np.mean(RMSEs_pos)
            mean_RMSE_vel = np.mean(RMSEs_vel)
            mean_lstm_losses = np.mean(lstm_losses)
            mean_actor_losses = np.mean(actor_losses)
            mean_critic_losses = np.mean(critic_losses)
            mean_state_values = np.mean(state_values)

            # Store results
            # For each timestep and agent
            # train_results[step]['SE_pos'] = SEs_pos
            # train_results[step]['SE_vel'] = SEs_vel
            # For each timestep
            train_results[step]['total_rewards'] = total_rewards
            train_results[step]['RMSE_pos'] = RMSEs_pos
            train_results[step]['RMSE_vel'] = RMSEs_vel
            train_results[step]['lstm_losses'] = lstm_losses
            train_results[step]['actor_losses'] = actor_losses
            train_results[step]['critic_losses'] = critic_losses
            train_results[step]['state_values'] = state_values

            if step % self.log_train_step == 0 and step != 0:
                self.save_model_and_train_result(step=step, train_results=train_results)

            print("episode_{} over\nMean reward {}, Mean state value {}\nMean RMSE pos {}, Mean RMSE vel {}\nMean LSTM loss {}, Mean Actor loss {}, Mean Critic loss {}\nMean Ratios {}, Mean Advantage function {}\n\n".\
                  format(step, mean_reward, mean_state_values, mean_RMSE_pos, mean_RMSE_vel, mean_lstm_losses, mean_actor_losses, mean_critic_losses, np.mean(ratios_mean), np.mean(advantage_function_mean)))

        return 



    def test(self, valid_numpy_dataset):

        valid_dict_dataset = {dataset_output_name:valid_numpy_dataset[index_name] for index_name,(dataset_output_name) in enumerate(self.env.dataset_output_names)}

        # reshape according to the sequence length
        valid_dict_dataset['t_A'] = valid_dict_dataset['t_A'].reshape(self.batch_size, self.L, self.num_agents, 4)
        valid_dict_dataset['x_GNSS'] = valid_dict_dataset['x_GNSS'].reshape(self.batch_size, self.L, self.num_agents, 4)
        valid_dict_dataset['x_A2A'] = valid_dict_dataset['x_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
        valid_dict_dataset['x_A2F'] = valid_dict_dataset['x_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)
        valid_dict_dataset['mutual_distances_A2A'] = valid_dict_dataset['mutual_distances_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
        valid_dict_dataset['mutual_distances_A2F'] = valid_dict_dataset['mutual_distances_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)
        valid_dict_dataset['connectivity_matrix_A2A'] = valid_dict_dataset['connectivity_matrix_A2A'].reshape(self.batch_size, self.L, self.num_agents, self.num_agents)
        valid_dict_dataset['connectivity_matrix_A2F'] = valid_dict_dataset['connectivity_matrix_A2F'].reshape(self.batch_size, self.L, self.num_agents, self.num_features)
        
        L = valid_dict_dataset['t_A'].shape[1]

        state = valid_dict_dataset['t_A'][:, 0].reshape(-1, 4)
        # No Cooperation
        no_coop_state_hat = deepcopy(state)
        no_coop_ppo_lstm_weights = deepcopy(self.agents.ppo_lstm.state_dict())
        # Cooperation
        state_hat = deepcopy(state)
        ppo_lstm_weights = deepcopy(self.agents.ppo_lstm.state_dict())

        # num_agents x (1 + num_agents + num_features)
        obs = np.concatenate((valid_dict_dataset['x_GNSS'][:, 0].reshape(-1, 4), 
                                valid_dict_dataset['x_A2A'][:, 0].reshape(-1, self.num_agents),
                                valid_dict_dataset['x_A2F'][:, 0].reshape(-1, self.num_features)), 1)
        
        # Subsitute nan values in x_A2A and x_A2F with -1:
        valid_dict_dataset['x_A2A'][np.where(np.isnan(valid_dict_dataset['x_A2A']))] = -1
        valid_dict_dataset['x_A2F'][np.where(np.isnan(valid_dict_dataset['x_A2F']))] = -1
        
        self.agents.ppo_lstm.init_hidden()

        # Cooperation
        hidden_lstm = deepcopy((self.agents.ppo_lstm.hidden_layer1[0].detach(), self.agents.ppo_lstm.hidden_layer1[1].detach()))
        hidden_lstm2 = deepcopy((self.agents.ppo_lstm.hidden_layer2[0].detach(), self.agents.ppo_lstm.hidden_layer2[1].detach()))

        # No Cooperation
        no_coop_hidden_lstm = deepcopy((self.agents.ppo_lstm.hidden_layer1[0].detach(), self.agents.ppo_lstm.hidden_layer1[1].detach()))
        no_coop_hidden_lstm2 = deepcopy((self.agents.ppo_lstm.hidden_layer2[0].detach(), self.agents.ppo_lstm.hidden_layer2[1].detach()))
        
        self.no_coop_agents = ICPMAPPO(self.params)
        mkdir(os.path.join(self.saved_models_dir, 'temp'))
        self.agents.save_model(saved_models_dir=os.path.join(self.saved_models_dir, 'temp'))
        self.no_coop_agents.load_model(saved_models_dir=os.path.join(self.saved_models_dir, 'temp'))
        self.agents.del_model(saved_models_dir=os.path.join(self.saved_models_dir, 'temp'))
        
        MARL_mean = []
        MARL_no_coop_mean = []
        MARL_absolute_error_pos = []
        MARL_absolute_error_vel = []
        MARL_no_coop_absolute_error_pos = []
        MARL_no_coop_absolute_error_vel = []

        total_rewards = []
        lstm_losses = []
        total_actions = []
        no_coop_total_rewards = []
        no_coop_lstm_losses = []
        no_coop_total_actions = []

        for t in range(1, L):

            connectivity_matrix_A2A = valid_dict_dataset['connectivity_matrix_A2A'][:, t].reshape(-1, self.num_agents)

            # Cooperation
            actions, log_probs, entropy = self.agents.choose_actions(hidden_lstm, train = 0, connectivity_matrix_A2A = connectivity_matrix_A2A, train = 0)
            # No Cooperation
            no_coop_actions, no_coop_log_probs, no_coop_entropy = self.no_coop_agents.choose_actions(no_coop_hidden_lstm, train = 0)

            obs_next = np.concatenate((valid_dict_dataset['x_GNSS'][:, t].reshape(-1, 4), 
                                valid_dict_dataset['x_A2A'][:, t].reshape(-1, self.num_agents),
                                valid_dict_dataset['x_A2F'][:, t].reshape(-1, self.num_features)), 1)
            
            state_next = valid_dict_dataset['t_A'][:, t].reshape(-1, 4)

            # Cooperation
            self.agents.ppo_lstm.hidden_layer2 = deepcopy((hidden_lstm2[0].detach(), hidden_lstm2[1].detach()))
            self.agents.ppo_lstm.load_state_dict(ppo_lstm_weights) 
            state_hat_next, hidden_lstm_next = self.agents.get_state_estimate(deepcopy(actions), deepcopy(hidden_lstm), deepcopy(obs_next), deepcopy(connectivity_matrix_A2A), train = 0)
            state_hat_next = state_hat_next.view(self.num_agents*self.batch_size, 4)
            SE_pos, RMSE_pos, SE_vel, RMSE_vel, lstm_loss = self.agents.learn_lstm(self.env, state_hat_next, deepcopy(state_next), train = 0)
            hidden_lstm2 = deepcopy((self.agents.ppo_lstm.hidden_layer2[0].detach(), self.agents.ppo_lstm.hidden_layer2[1].detach()))
            ppo_lstm_weights = deepcopy(self.agents.ppo_lstm.state_dict())
            rewards = self.agents.get_reward(self.env, state, state_hat, state_next, state_hat_next)

            # No Cooperation
            self.no_coop_agents.ppo_lstm.hidden_layer2 = deepcopy((no_coop_hidden_lstm2[0].detach(), no_coop_hidden_lstm2[1].detach()))
            self.no_coop_agents.ppo_lstm.load_state_dict(no_coop_ppo_lstm_weights)
            no_coop_state_hat_next, no_coop_hidden_lstm_next = self.no_coop_agents.get_state_estimate_single_agent(deepcopy(no_coop_actions), no_coop_hidden_lstm, deepcopy(obs_next), deepcopy(connectivity_matrix_A2A), train = 0)
            no_coop_state_hat_next = no_coop_state_hat_next.view(self.num_agents*self.batch_size, 4)
            no_coop_SE_pos, no_coop_RMSE_pos, no_coop_SE_vel, no_coop_RMSE_vel, no_coop_lstm_loss = self.no_coop_agents.learn_lstm(self.env, no_coop_state_hat_next, deepcopy(state_next), train = 0)
            no_coop_hidden_lstm2 = deepcopy((self.no_coop_agents.ppo_lstm.hidden_layer2[0].detach(), self.no_coop_agents.ppo_lstm.hidden_layer2[1].detach()))
            no_coop_ppo_lstm_weights = deepcopy(self.no_coop_agents.ppo_lstm.state_dict())
            no_coop_rewards = self.no_coop_agents.get_reward(self.env, state, state_hat, state_next, state_hat_next)

            # Next timestep
            total_rewards.append(rewards)
            no_coop_total_rewards.append(no_coop_rewards)
            state = state_next
            state_hat = state_hat_next
            no_coop_state_hat = no_coop_state_hat_next
            obs = obs_next
            hidden_lstm = (hidden_lstm_next[0].detach(), hidden_lstm_next[1].detach())
            no_coop_hidden_lstm = (no_coop_hidden_lstm_next[0].detach(), no_coop_hidden_lstm_next[1].detach())

            # Store results
            state_hat_original_space = deepcopy(state_hat.detach())
            state_hat_original_space[:,0:2] = normalize_numpy(state_hat[:,0:2].detach(), np.array([self.env.limit_pos1[0], self.env.limit_pos2[0]]), np.array([self.env.limit_pos1[1], self.env.limit_pos2[1]]), normalize = 0, type_='minmax', already_centered = 0)
            state_hat_original_space[:,2:4] = normalize_numpy(state_hat[:,2:4].detach(), np.array([self.env.limit_vel1[0], self.env.limit_vel2[0]]), np.array([self.env.limit_vel1[1], self.env.limit_vel2[1]]), normalize = 0, type_='minmax', already_centered = 0)
            
            no_coop_state_hat_original_space = deepcopy(no_coop_state_hat.detach())
            no_coop_state_hat_original_space[:,0:2] = normalize_numpy(no_coop_state_hat[:,0:2].detach(), np.array([self.env.limit_pos1[0], self.env.limit_pos2[0]]), np.array([self.env.limit_pos1[1], self.env.limit_pos2[1]]), normalize = 0, type_='minmax', already_centered = 0)
            no_coop_state_hat_original_space[:,2:4] = normalize_numpy(no_coop_state_hat[:,2:4].detach(), np.array([self.env.limit_vel1[0], self.env.limit_vel2[0]]), np.array([self.env.limit_vel1[1], self.env.limit_vel2[1]]), normalize = 0, type_='minmax', already_centered = 0)

            MARL_mean.append(state_hat_original_space.cpu().numpy())
            MARL_absolute_error_pos.append(np.sqrt(SE_pos.cpu()).numpy())
            MARL_absolute_error_vel.append(np.sqrt(SE_vel.cpu()).numpy())
            lstm_losses.append(lstm_loss.cpu().numpy())
            total_actions.append(actions)

            MARL_no_coop_mean.append(no_coop_state_hat_original_space.cpu().numpy())
            MARL_no_coop_absolute_error_pos.append(np.sqrt(no_coop_SE_pos.cpu()).numpy())
            MARL_no_coop_absolute_error_vel.append(np.sqrt(no_coop_SE_vel.cpu()).numpy())
            no_coop_lstm_losses.append(lstm_loss.cpu().numpy())
            no_coop_total_actions.append(no_coop_actions)


       #####
    

        num_samples = len(MARL_mean)
        results = {
                   'MARL_mean':np.array(MARL_mean).reshape(-1, self.num_agents, 4), # (timesteps, num_agents, 4)
                   'MARL_no_coop_mean': np.array(MARL_no_coop_mean).reshape(-1, self.num_agents, 4),  # (timesteps, num_agents, 4)
            
                   'MARL_absolute_error_pos': np.array(MARL_absolute_error_pos).reshape(-1, self.num_agents), # (timesteps, num_agents)
                   'MARL_no_coop_absolute_error_pos': np.array(MARL_no_coop_absolute_error_pos).reshape(-1, self.num_agents), # (timesteps, num_agents)

                   'total_actions':np.sum(np.array(total_actions).reshape(-1, self.num_agents, self.num_agents), 2), # (timesteps, num_agents, 1)
                    }

        return results

    def save_model_and_train_result(self, step = None, train_results = None, saved_models_dir = None):
        self.agents.save_model_and_train_result(step = step, train_results = train_results, saved_models_dir = saved_models_dir)

    def load_model_and_train_result(self, saved_models_dir = None):
        return self.agents.load_model_and_train_result(saved_models_dir = saved_models_dir)
    
    def save_test_result(self, test_results):
        self.agents.save_test_result(test_results)

    def load_test_result(self):
        return self.agents.load_test_result()

    def set_cuda_device(self, use_cuda = None):
        self.agents.set_cuda_device(use_cuda)

        if use_cuda == None:
            use_cuda = self.use_cuda
        
        if use_cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')