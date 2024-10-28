import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy
from copy import copy
import math
import re

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
CODE_DIR = os.path.split(os.path.split(cwd)[0])[0]
CLASSES_DIR = os.path.join(CODE_DIR, 'Classes')
EXPERIMENTS_DIR = os.path.join(CODE_DIR, 'Exp')
DB_DIR = os.path.join(os.path.split(CODE_DIR)[0], 'DB')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(CODE_DIR))
from Classes.utils.utils import print_name_and_value, normalize_numpy, mkdir

class DATASET(object):

    DEFAULTS = {}   

    def __init__(self, params = {}):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(DATASET.DEFAULTS, **params_dict)

    ###########################################################################################
    ###########################################################################################
    # RETURN DATASETS

    def return_dataset_tracking(self, train = 1):

        self.DB_path = os.path.join(DB_DIR, self.DB_name)
        self.DB_path_mat = os.path.join(self.DB_path, 'dataset.mat')
        self.DB_path_pt = os.path.join(self.DB_path, 'dataset.pt')

        # Timestep
        self.n = 0

        # Type of noise
        if self.noise_type == 'Gaussian':
            self.noise_class = np.random.normal
        elif self.noise_type == 'Laplacian':
            self.noise_class = np.random.laplace
        else:
            self.noise_class = np.random.normal

        # Motion evolution matrices
        # x_n = F x_n-1 + W w_n-1
        # x: pos, vel, acc. w: pos_noise, vel_noise, acc_noise
        self.F = np.array([[1, 0, self.T_between_timestep, 0                      , (self.T_between_timestep**2)/2, 0                             ],
                           [0, 1, 0                      , self.T_between_timestep, 0                             , (self.T_between_timestep**2)/2],
                           [0, 0, 1                      , 0                      , self.T_between_timestep       , 0                             ],
                           [0, 0, 0                      , 1                      , 0                             , self.T_between_timestep       ],
                           [0, 0, 0                      , 0                      , 1                             , 0                             ],
                           [0, 0, 0                      , 0                      , 0                             , 1                             ]])
        self.W = copy(self.F)


        # Recompute dataset
        if not self.bool_load_dataset or not os.path.exists(self.DB_path_pt):

            mkdir(self.DB_path)
            
            # Compute initial positions, velocities and accelerations of each agent and feature
            self.define_initial_t()

            # Compute initial distances
            # timesteps x self.num_agents x self.num_agents
            self.mutual_distances_A2A = [scipy.spatial.distance.cdist(np.array([self.positions[f'A{agent}'] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[f'A{agent}'] for agent in range(self.num_agents)]).squeeze()).tolist()]
            # timesteps x self.num_agents x self.num_features
            self.mutual_distances_A2F = [scipy.spatial.distance.cdist(np.array([self.positions[f'A{agent}'] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[f'F{feature}'] for feature in range(self.num_features)]).squeeze()).tolist()]
            
            # Connectivity matrix based on communication distances
            # timesteps x self.num_agents x self.num_agents
            self.connectivity_matrix_A2A = [(np.array(self.mutual_distances_A2A)<self.comm_distance).squeeze()*1 - np.eye(self.num_agents)]
            # # timesteps x self.num_agents x self.num_features
            self.connectivity_matrix_A2F = [(np.array(self.mutual_distances_A2F)<self.comm_distance).squeeze()*1]

            # GT 
            # self.num_agents x timesteps x num_inputs (6)
            self.t_A = {f'A{agent}':[[self.positions[f'A{agent}'][self.n][0], 
                            self.positions[f'A{agent}'][self.n][1], 
                            self.velocities[f'A{agent}'][self.n][0], 
                            self.velocities[f'A{agent}'][self.n][1], 
                            self.accelerations[f'A{agent}'][self.n][0], 
                            self.accelerations[f'A{agent}'][self.n][1]]] for agent in range(self.num_agents)}
            # self.num_features x timesteps x num_inputs (6)
            self.t_F = {f'F{feature}':[[self.positions[f'F{feature}'][self.n][0], 
                            self.positions[f'F{feature}'][self.n][1], 
                            self.velocities[f'F{feature}'][self.n][0], 
                            self.velocities[f'F{feature}'][self.n][1], 
                            self.accelerations[f'F{feature}'][self.n][0], 
                            self.accelerations[f'F{feature}'][self.n][1]]] for feature in range(self.num_features)}
            # Input features  
            # self.num_agents x timesteps x self.num_agents
            self.x_A2A = {f'A{agent}':[np.abs(np.array(self.mutual_distances_A2A[0][agent]) + self.noise_class(0, self.meas_A2A_std_dist, (self.num_agents,)))] for agent in range(self.num_agents)}
            # self.num_agents x timesteps x self.num_features
            self.x_A2F = {f'A{agent}':[np.abs(np.array(self.mutual_distances_A2F[0][agent]) + self.noise_class(0, self.meas_A2F_std_dist, (self.num_features,)))]  for agent in range(self.num_agents)}
            # self.num_agents x timesteps x num_inputs (6)
            self.x_GNSS = {f'A{agent}':[[self.positions[f'A{agent}'][0][0] + self.noise_class(0, self.meas_GNSS_std_pos, (1,)).item(), 
                                self.positions[f'A{agent}'][0][1] + self.noise_class(0, self.meas_GNSS_std_pos, (1,)).item(), 
                                self.velocities[f'A{agent}'][0][0] + self.noise_class(0, self.meas_GNSS_std_vel, (1,)).item(), 
                                self.velocities[f'A{agent}'][0][1] + self.noise_class(0, self.meas_GNSS_std_vel, (1,)).item(), 
                                self.accelerations[f'A{agent}'][0][0] + self.noise_class(0, self.meas_GNSS_std_acc, (1,)).item(), 
                                self.accelerations[f'A{agent}'][0][1] + self.noise_class(0, self.meas_GNSS_std_acc, (1,)).item()]] for agent in range(self.num_agents)}
            
            # Compute dataset
            for n in range(self.H):

                self.compute_next_step()

            # Create torch dataloader (divided into sequence length L)
            # timesteps x self.num_agents x num_inputs (6)
            self.t_A = np.swapaxes(np.array(np.array([v for k,v in self.t_A.items()])), 0, 1)
            # timesteps x self.num_features x num_inputs (6)
            self.t_F = np.swapaxes(np.array(np.array([v for k,v in self.t_F.items()])), 0, 1)
            # timesteps x self.num_agents x self.num_agents
            self.x_A2A = np.swapaxes(np.array(np.array([v for k,v in self.x_A2A.items()])), 0, 1)
            # timesteps x self.num_agents x self.num_features
            self.x_A2F = np.swapaxes(np.array(np.array([v for k,v in self.x_A2F.items()])), 0, 1)
            # timesteps x self.num_agents x num_inputs (6)
            self.x_GNSS = np.swapaxes(np.array(np.array([v for k,v in self.x_GNSS.items()])), 0, 1)

        # Load dataset
        else:
            dataset = torch.load(self.DB_path_pt)

        if not self.bool_load_dataset or not os.path.exists(self.DB_path_pt):

            # Normalize (max-min)
            self.t_A[:,:,0:2] = normalize_numpy(self.t_A[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
            self.t_A[:,:,2:4] = normalize_numpy(self.t_A[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')
            self.t_A[:,:,4:6] = normalize_numpy(self.t_A[:,:,4:6], np.array([self.limit_acc1[0], self.limit_acc2[0]]), np.array([self.limit_acc1[1], self.limit_acc2[1]]), normalize = 1, type_='minmax')

            self.t_F[:,:,0:2] = normalize_numpy(self.t_F[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
            self.t_F[:,:,2:4] = normalize_numpy(self.t_F[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')
            self.t_F[:,:,4:6] = normalize_numpy(self.t_F[:,:,4:6], np.array([self.limit_acc1[0], self.limit_acc2[0]]), np.array([self.limit_acc1[1], self.limit_acc2[1]]), normalize = 1, type_='minmax')

            self.x_A2A = normalize_numpy(self.x_A2A, 0, np.linalg.norm([np.diff(self.limit_pos1), np.diff(self.limit_pos2)]), normalize = 1, type_='minmax')
            self.x_A2F = normalize_numpy(self.x_A2F, 0, np.linalg.norm([np.diff(self.limit_pos1), np.diff(self.limit_pos2)]), normalize = 1, type_='minmax')
            self.x_GNSS[:,:,0:2] = normalize_numpy(self.x_GNSS[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
            self.x_GNSS[:,:,2:4] = normalize_numpy(self.x_GNSS[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')
            self.x_GNSS[:,:,4:6] = normalize_numpy(self.x_GNSS[:,:,4:6], np.array([self.limit_acc1[0], self.limit_acc2[0]]), np.array([self.limit_acc1[1], self.limit_acc2[1]]), normalize = 1, type_='minmax')

            dataset = TensorDataset(torch.tensor(self.t_A).float(), 
                                        torch.tensor(self.t_F).float(), 
                                        torch.tensor(self.x_A2A).float(), 
                                        torch.tensor(self.x_GNSS).float(), 

                                        torch.tensor(self.mutual_distances_A2A).float(), 
                                        torch.tensor(self.mutual_distances_A2F).float(), 
                                        torch.tensor(self.connectivity_matrix_A2A).float(), 
                                        torch.tensor(self.connectivity_matrix_A2F).float(),  
                                        )    
            # Delete variables
            del self.positions; del self.velocities; del self.accelerations;
            del self.t_A; del self.t_F; del self.x_A2A; del self.x_A2F; del self.x_GNSS;
            del self.mutual_distances_A2A; del self.mutual_distances_A2F; del self.connectivity_matrix_A2A; del self.connectivity_matrix_A2F

        if self.bool_save_dataset:
            torch.save(dataset, self.DB_path_pt)

        self.dataset_output_names = ['t_A', 't_F', 'x_A2A', 'x_A2F', 'x_GNSS', 
                                     'mutual_distances_A2A', 'mutual_distances_A2F', 'connectivity_matrix_A2A', 'connectivity_matrix_A2F']

        if train:
            loader = DataLoader(dataset, batch_size=self.L, shuffle=1, pin_memory=True)
            self.params.num_train_batches = len(loader)
            print_name_and_value(self.params.num_train_batches)
        else:
            loader = DataLoader(dataset, batch_size=self.L, shuffle=0, pin_memory=True)
            self.params.num_valid_batches = len(loader)
            print_name_and_value(self.params.num_valid_batches)
        
        self.params.dataset_output_names = self.dataset_output_names

        self.params.update_all()

        return loader


    ###########################################################################################
    ###########################################################################################
    # SHOW DATASETS

    def show_dataset(self, loader = None):
        # TODO
        pass
    def show_dataset_tracking(self):
        # TODO
        pass
    def show_tracking_results(self, output_results_tracking):
        # TODO
        pass


    ###########################################################################################
    ###########################################################################################
    # AUXILIRAY
        
    def compute_next_step(self): 

        # Agents
        for agent in range(self.num_agents):

            noise_position, noise_velocities, noise_accelerations = self.add_noise_deterministic_components(f'A{agent}', 'agent')
            new_t = self.F@np.array(self.t_A[f'A{agent}'][-1]) + self.W@np.concatenate((noise_position, noise_velocities, noise_accelerations))
            new_t_list = new_t.tolist()

            # Check if agent is outside the area
            if self.limit_behavior == 'reflection':
                if new_t_list[0] < self.limit_pos1[0]:
                    while new_t_list[0] < self.limit_pos1[0]:
                        new_t_list[0] = self.limit_pos1[0] + abs(self.limit_pos1[0]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                elif new_t_list[0] > self.limit_pos1[1]:
                    while new_t_list[0] > self.limit_pos1[1]:
                        new_t_list[0] = self.limit_pos1[1] - abs(self.limit_pos1[1]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                if new_t_list[1] < self.limit_pos2[0]:
                    while new_t_list[1] < self.limit_pos2[0]:
                        new_t_list[1] = self.limit_pos2[0] + abs(self.limit_pos2[0]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
                elif new_t_list[1] > self.limit_pos2[1]:
                    while new_t_list[1] > self.limit_pos2[1]:
                        new_t_list[1] = self.limit_pos2[1] - abs(self.limit_pos2[1]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
            elif self.limit_behavior == 'continue':
                if new_t_list[0] < self.limit_pos1[0]:
                    while new_t_list[0] < self.limit_pos1[0]:
                        new_t_list[0] = self.limit_pos1[1] - abs(self.limit_pos1[0]-new_t_list[0])
                if new_t_list[0] > self.limit_pos1[1]:
                    while new_t_list[0] > self.limit_pos1[1]:
                        new_t_list[0] = self.limit_pos1[0] + abs(self.limit_pos1[1]-new_t_list[0])
                if new_t_list[1] < self.limit_pos2[0]:
                    while new_t_list[1] < self.limit_pos2[0]:
                        new_t_list[1] = self.limit_pos2[1] - abs(self.limit_pos2[0]-new_t_list[1])
                if new_t_list[1] > self.limit_pos2[1]:
                    while new_t_list[1] > self.limit_pos2[1]:
                        new_t_list[1] = self.limit_pos2[0] + abs(self.limit_pos2[1]-new_t_list[1])

            # Limit velocity
            if new_t_list[2] < self.limit_vel1[0]:
                new_t_list[2] = self.limit_vel1[0]
            if new_t_list[2] > self.limit_vel1[1]:
                new_t_list[2] = self.limit_vel1[1]
        
            if new_t_list[3] < self.limit_vel2[0]:
                new_t_list[3] = self.limit_vel2[0]
            if new_t_list[3] > self.limit_vel2[1]:
                new_t_list[3] = self.limit_vel2[1]

            # Limit acceleration
            if new_t_list[4] < self.limit_acc1[0]:
                new_t_list[4] = self.limit_acc1[0]
            if new_t_list[4] > self.limit_acc1[1]:
                new_t_list[4] = self.limit_acc1[1]
        
            if new_t_list[4] < self.limit_acc2[0]:
                new_t_list[4] = self.limit_acc2[0]
            if new_t_list[4] > self.limit_acc2[1]:
                new_t_list[4] = self.limit_acc2[1]

            self.t_A[f'A{agent}'].append(new_t_list)
            self.positions[f'A{agent}'].append(new_t_list[0:2])
            self.velocities[f'A{agent}'].append(new_t_list[2:4])
            self.accelerations[f'A{agent}'].append(new_t_list[4:6])

            self.x_GNSS[f'A{agent}'].append([self.positions[f'A{agent}'][-1][0] + self.noise_class(0, self.meas_GNSS_std_pos, (1,)).item(), 
                                    self.positions[f'A{agent}'][-1][1] + self.noise_class(0, self.meas_GNSS_std_pos, (1,)).item(), 
                                    self.velocities[f'A{agent}'][-1][0] + self.noise_class(0, self.meas_GNSS_std_vel, (1,)).item(), 
                                    self.velocities[f'A{agent}'][-1][1] + self.noise_class(0, self.meas_GNSS_std_vel, (1,)).item(), 
                                    self.accelerations[f'A{agent}'][-1][0] + self.noise_class(0, self.meas_GNSS_std_acc, (1,)).item(), 
                                    self.accelerations[f'A{agent}'][-1][1] + self.noise_class(0, self.meas_GNSS_std_acc, (1,)).item()])

        # Features
        for feature in range(self.num_features):

            noise_position, noise_velocities, noise_accelerations = self.add_noise_deterministic_components(f'F{feature}', 'feature')
            new_t = self.F@np.array(self.t_F[f'F{feature}'][-1]) + self.W@np.concatenate((noise_position, noise_velocities, noise_accelerations))
            new_t_list = new_t.tolist()

            # Check if feature is outside the area
            if self.limit_behavior == 'reflection':
                if new_t_list[0] < self.limit_pos1[0]:
                    while new_t_list[0] < self.limit_pos1[0]:
                        new_t_list[0] = self.limit_pos1[0] + abs(self.limit_pos1[0]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                elif new_t_list[0] > self.limit_pos1[1]:
                    while new_t_list[0] > self.limit_pos1[1]:
                        new_t_list[0] = self.limit_pos1[1] - abs(self.limit_pos1[1]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                if new_t_list[1] < self.limit_pos2[0]:
                    while new_t_list[1] < self.limit_pos2[0]:
                        new_t_list[1] = self.limit_pos2[0] + abs(self.limit_pos2[0]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
                elif new_t_list[1] > self.limit_pos2[1]:
                    while new_t_list[1] > self.limit_pos2[1]:
                        new_t_list[1] = self.limit_pos2[1] - abs(self.limit_pos2[1]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
            elif self.limit_behavior == 'continue':
                if new_t_list[0] < self.limit_pos1[0]:
                    while new_t_list[0] < self.limit_pos1[0]:
                        new_t_list[0] = self.limit_pos1[1] - abs(self.limit_pos1[0]-new_t_list[0])
                if new_t_list[0] > self.limit_pos1[1]:
                    while new_t_list[0] > self.limit_pos1[1]:
                        new_t_list[0] = self.limit_pos1[0] + abs(self.limit_pos1[1]-new_t_list[0])
                if new_t_list[1] < self.limit_pos2[0]:
                    while new_t_list[1] < self.limit_pos2[0]:
                        new_t_list[1] = self.limit_pos2[1] - abs(self.limit_pos2[0]-new_t_list[1])
                if new_t_list[1] > self.limit_pos2[1]:
                    while new_t_list[1] > self.limit_pos2[1]:
                        new_t_list[1] = self.limit_pos2[0] + abs(self.limit_pos2[1]-new_t_list[1])

            # Limit velocity
            if new_t_list[2] < self.limit_vel1[0]:
                new_t_list[2] = self.limit_vel1[0]
            if new_t_list[2] > self.limit_vel1[1]:
                new_t_list[2] = self.limit_vel1[1]
        
            if new_t_list[3] < self.limit_vel2[0]:
                new_t_list[3] = self.limit_vel2[0]
            if new_t_list[3] > self.limit_vel2[1]:
                new_t_list[3] = self.limit_vel2[1]

            # Limit acceleration
            if new_t_list[4] < self.limit_acc1[0]:
                new_t_list[4] = self.limit_acc1[0]
            if new_t_list[4] > self.limit_acc1[1]:
                new_t_list[4] = self.limit_acc1[1]
        
            if new_t_list[4] < self.limit_acc2[0]:
                new_t_list[4] = self.limit_acc2[0]
            if new_t_list[4] > self.limit_acc2[1]:
                new_t_list[4] = self.limit_acc2[1]

            self.t_F[f'F{feature}'].append(new_t_list)
            self.positions[f'F{feature}'].append(new_t_list[0:2])
            self.velocities[f'F{feature}'].append(new_t_list[2:4])
            self.accelerations[f'F{feature}'].append(new_t_list[4:6])

        # Compute real distances
        self.mutual_distances_A2A.append(scipy.spatial.distance.cdist(np.array([self.positions[f'A{agent}'][-1] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[f'A{agent}'][-1] for agent in range(self.num_agents)]).squeeze()).tolist())
        self.mutual_distances_A2F.append(scipy.spatial.distance.cdist(np.array([self.positions[f'A{agent}'][-1] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[f'F{feature}'][-1] for feature in range(self.num_features)]).squeeze()).tolist())

        # Compute connectivity matrix
        self.connectivity_matrix_A2A.append((np.array(self.mutual_distances_A2A[-1])<self.comm_distance).squeeze()*1 - np.eye(self.num_agents))
        self.connectivity_matrix_A2F.append((np.array(self.mutual_distances_A2F[-1])<self.comm_distance).squeeze()*1)

        # Compute inter-distances measurements
        for agent in range(self.num_agents):
            self.x_A2A[f'A{agent}'].append(np.abs(np.array(self.mutual_distances_A2A[-1][agent]) + self.noise_class(0, self.meas_A2A_std_dist, (self.num_agents,))))
            self.x_A2F[f'A{agent}'].append(np.abs(np.array(self.mutual_distances_A2F[-1][agent]) + self.noise_class(0, self.meas_A2F_std_dist, (self.num_features,))))

        # Update timestep
        self.n = self.n + 1


    def add_noise_deterministic_components(self, agent_or_feature, type_):

        if type_ == 'agent':
            # Random component
            noise_position = self.noise_class(0, self.motion_A_std_pos, 2) 
            noise_velocities = self.noise_class(0, self.motion_A_std_vel, 2) 
            noise_accelerations = self.noise_class(0, self.motion_A_std_acc, 2) 
        elif type_ == 'feature':
            # Random component
            noise_position = self.noise_class(0, self.motion_F_std_pos, 2) 
            noise_velocities = self.noise_class(0, self.motion_F_std_vel, 2) 
            noise_accelerations = self.noise_class(0, self.motion_F_std_acc, 2) 
        # Deterministic component
        if type_ == 'agent':
            setting_trajectories = self.setting_trajectories_A
        elif type_ == 'feature':
            setting_trajectories = self.setting_trajectories_F
        if setting_trajectories == 'spiral' and self.n >3: # self.n >20:

            # Perfect spiral
            # angle = np.arctan2(self.positions[agent_or_feature][-1][1], self.positions[agent_or_feature][-1][0])
            # angle_deg = angle*180/np.pi
            # distance_from_center = np.linalg.norm([self.positions[agent_or_feature][-1][0],self.positions[agent_or_feature][-1][1]])
            # if type == 'agent':
            #     mod_vel = np.max((np.linalg.norm([self.initial_A_mean_vel1, self.initial_A_mean_vel2]), 1))
            # elif type == 'feature':
            #     mod_vel = np.max((np.linalg.norm([self.initial_F_mean_vel1, self.initial_F_mean_vel2]), 1))
            # orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_pos1[0])
            # # remove velocity
            # orthogonal_vel -= np.array([self.velocities[agent_or_feature][-1][0],self.velocities[agent_or_feature][-1][1]])
            # noise_velocities += orthogonal_vel

            # Golden spiral
            angle = np.arctan2(self.positions[agent_or_feature][-1][1], self.positions[agent_or_feature][-1][0])
            angle_deg = angle*180/np.pi
            distance_from_center = np.linalg.norm([self.positions[agent_or_feature][-1][0],self.positions[agent_or_feature][-1][1]])
            if type_ == 'agent':
                mod_vel = np.max((np.linalg.norm([self.initial_A_mean_vel1, self.initial_A_mean_vel2]), 1))
            elif type_ == 'feature':
                mod_vel = np.max((np.linalg.norm([self.initial_F_mean_vel1, self.initial_F_mean_vel2]), 1))
            orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_pos1[0])

            if self.n + int(re.findall(r'\d+', agent_or_feature)) > 10 + 16:
                # Remove velocity
                orthogonal_vel -= np.array([self.velocities[agent_or_feature][-1][0],self.velocities[agent_or_feature][-1][1]])
            noise_velocities += orthogonal_vel

        return noise_position, noise_velocities, noise_accelerations

    def define_initial_t(self):
    
        # Agents
        if self.setting_trajectories_A == 'not_defined':
            mean1 = self.initial_A_mean_pos1; mean2 = self.initial_A_mean_pos2
            std1 = self.initial_A_std_pos1; std2 = self.initial_A_std_pos2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.positions = {f'A{agent}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for agent in range(self.num_agents)}
            mean1 = self.initial_A_mean_vel1; mean2 = self.initial_A_mean_vel2
            std1 = self.initial_A_std_vel1; std2 = self.initial_A_std_vel2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.velocities = {f'A{agent}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for agent in range(self.num_agents)}
            mean1 = self.initial_A_mean_acc1; mean2 = self.initial_A_mean_acc2
            std1 = self.initial_A_std_acc1; std2 = self.initial_A_std_acc2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.accelerations = {f'A{agent}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for agent in range(self.num_agents)}
        elif self.setting_trajectories == 'star' or self.setting_trajectories == 'spiral':
            angle_directions = np.arange(0,360, 360/self.num_agents) * math.pi/180
            mean1 = self.initial_A_mean_pos1; mean2 = self.initial_A_mean_pos2
            std1 = self.initial_A_std_pos1; std2 = self.initial_A_std_pos2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.positions = {f'A{agent}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for agent in range(self.num_agents)}
            self.velocities = {f'A{agent}':[[abs(self.initial_A_mean_vel1)*np.cos(angle_directions[agent]), abs(self.initial_A_mean_vel1)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}
            self.accelerations = {f'A{agent}':[[abs(self.initial_A_mean_acc1)*np.cos(angle_directions[agent]), abs(self.initial_A_mean_acc1)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}    

        # Features
        if self.setting_trajectories_F == 'not_defined':
            mean1 = self.initial_F_mean_pos1; mean2 = self.initial_F_mean_pos2
            std1 = self.initial_F_std_pos1; std2 = self.initial_F_std_pos2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.positions = {**self.positions, **{f'F{feature}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for feature in range(self.num_features)}}
            mean1 = self.initial_F_mean_vel1; mean2 = self.initial_F_mean_vel2
            std1 = self.initial_F_std_vel1; std2 = self.initial_F_std_vel2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.velocities = {**self.velocities, **{f'F{feature}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for feature in range(self.num_features)}}
            mean1 = self.initial_F_mean_acc1; mean2 = self.initial_F_mean_acc2
            std1 = self.initial_F_std_acc1; std2 = self.initial_F_std_acc2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.accelerations = {**self.accelerations, **{f'F{feature}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for feature in range(self.num_features)}}
        elif self.setting_trajectories == 'star' or self.setting_trajectories == 'spiral':
            angle_directions = np.arange(0,360, 360/self.num_features) * math.pi/180
            mean1 = self.initial_F_mean_pos1; mean2 = self.initial_F_mean_pos2
            std1 = self.initial_F_std_pos1; std2 = self.initial_F_std_pos2
            a1 = mean1 - np.sqrt(3) * std1; b1 = mean1 + np.sqrt(3) * std1
            a2 = mean2 - np.sqrt(3) * std2; b2 = mean2 + np.sqrt(3) * std2
            self.positions = {**self.positions, **{f'F{feature}':[[np.random.uniform(a1,b1), np.random.uniform(a2,b2)]] for feature in range(self.num_features)}}
            self.velocities = {**self.velocities, **{f'F{feature}':[[abs(self.initial_F_mean_vel1)*np.cos(angle_directions[feature]), abs(self.initial_F_mean_vel1)*np.sin(angle_directions[feature])]] for feature in range(self.num_features)}}
            self.accelerations = {**self.accelerations, **{f'F{feature}':[[abs(self.initial_F_mean_acc1)*np.cos(angle_directions[feature]), abs(self.initial_F_mean_acc1)*np.sin(angle_directions[feature])]] for feature in range(self.num_features)}}    

