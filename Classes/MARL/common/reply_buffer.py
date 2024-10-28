import threading
import torch
from numpy import ndarray
import numpy as np


LOCK_MEMORY = threading.Lock()

class CommBatchEpisodeMemory(object):

    def __init__(self, continuous_actions: bool = False, n_actions: int = 0, n_agents: int = 0):
        self.continuous_actions = continuous_actions
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.obs = []
        self.obs_next = []
        self.state = []
        self.state_hat = []
        self.state_next = []
        self.state_hat_next = []
        self.rewards = []
        self.unit_actions = []
        self.log_probs = []
        self.unit_actions_onehot = []
        self.per_episode_len = []
        self.hidden_lstm = []
        self.n_step = 0

    def store_one_episode(self, one_obs: dict, one_state: ndarray, one_state_hat: ndarray, action: list, reward: float,
                          one_obs_next: dict = None, one_state_next: ndarray = None, one_state_hat_next: ndarray = None, 
                          log_probs: list = None, hidden_lstm = None):
        self.obs.append(torch.Tensor(one_obs))
        self.state.append(torch.Tensor(one_state))
        self.state_hat.append(torch.Tensor(one_state_hat.cpu()))
        self.rewards.append(torch.Tensor(reward))
        self.unit_actions.append(action)
        if one_obs_next is not None:
            self.obs_next.append(torch.Tensor(one_obs_next))
        if one_state_next is not None:
            self.state_next.append(torch.Tensor(one_state_next))
        if one_state_hat_next is not None:
            self.state_hat_next.append(torch.Tensor(one_state_hat_next.cpu()))
        if log_probs is not None:
            self.log_probs.append(log_probs)
        if hidden_lstm is not None:
            self.hidden_lstm.append(hidden_lstm)
        self.n_step += 1

    def clear_memories(self):
        self.obs.clear()
        self.obs_next.clear()
        self.state.clear()
        self.state_hat.clear()
        self.state_next.clear()
        self.state_hat_next.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.unit_actions.clear()
        self.unit_actions_onehot.clear()
        self.per_episode_len.clear()
        self.hidden_lstm.clear()
        self.n_step = 0

    def set_per_episode_len(self, episode_len: int):
        self.per_episode_len.append(episode_len)

    def get_batch_data(self) -> dict:

        obs = torch.stack(self.obs, dim=0)
        state = torch.stack(self.state, dim=0)
        state_hat = torch.stack(self.state_hat, dim=0)
        rewards = torch.stack(self.rewards, dim=0) # reshape_tensor_from_list(torch.Tensor(self.rewards), self.per_episode_len)
        actions = torch.Tensor(np.array(self.unit_actions))
        obs_next = torch.stack(self.obs_next, dim=0)
        state_next = torch.stack(self.state_next, dim=0)
        state_hat_next = torch.stack(self.state_hat_next, dim=0)
        log_probs = torch.stack(self.log_probs, dim=0)
        hidden_lstm = self.hidden_lstm
        data = {
            'obs': obs,
            'state': state,
            'state_hat': state_hat,
            'rewards': rewards,
            'actions': actions,
            'obs_next': obs_next,
            'state_next': state_next, 
            'state_hat_next': state_hat_next,  
            'hidden_lstm': hidden_lstm,  

            'log_probs': log_probs,
            'per_episode_len': self.per_episode_len
        }
        return data


