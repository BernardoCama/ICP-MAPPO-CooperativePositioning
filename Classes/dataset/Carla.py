import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import scipy
from copy import copy, deepcopy
import mat73

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

    def return_dataset_tracking(self, train = 1, batch_size = None, normalize = 1, shuffle = 0, 
                                bool_save_dataset = None, bool_load_dataset = None, specific_dataset_name = 'Town02'):

        if train:
            specific_dataset_name = specific_dataset_name
        else:
            specific_dataset_name = specific_dataset_name # 'Town10' or'Town02' 
        self.__dict__.update(DATASET.DEFAULTS, **self.limit[specific_dataset_name])

        if batch_size == None:
            batch_size = self.L
        if bool_save_dataset == None:
            bool_save_dataset = self.bool_save_dataset
        if bool_load_dataset == None:
            bool_load_dataset = self.bool_load_dataset

        self.DB_path = os.path.join(DB_DIR, self.DB_name, specific_dataset_name) 
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
        if not bool_load_dataset or not os.path.exists(self.DB_path_pt):

            mkdir(self.DB_path)
            
            # import datasets
            try:
                raw_dataset = mat73.loadmat(self.DB_path_mat)
            except: 
                raw_dataset = scipy.io.loadmat(self.DB_path_mat)
            
            raw_dataset_original = deepcopy(raw_dataset)
            print('Loaded dataset')

            self.num_features, _, _, self.H, self.num_agents = raw_dataset['Fea_vect_boxes_oracle'].shape

            # timesteps x self.num_agents x num_inputs (4)
            raw_dataset['vehicleStateGT'] = raw_dataset['vehicleStateGT'].reshape(-1, 4, self.H)
            self.t_A = np.swapaxes(np.swapaxes(raw_dataset['vehicleStateGT'], 0, 2), 1, 2)
            # timesteps x self.num_features x num_inputs (4)
            raw_dataset['Fea_vect_true'] = raw_dataset['Fea_vect_true'].reshape(-1, 4, self.H)
            self.t_F = np.swapaxes(np.swapaxes(raw_dataset['Fea_vect_true'], 0, 2), 1, 2)

            # timesteps x self.num_agents x self.num_agents
            self.mutual_distances_A2A = np.array([[scipy.spatial.distance.cdist(np.array([self.t_A[n,agent,0:2].squeeze() for agent in range(self.num_agents)]).squeeze(),np.array([self.t_A[n,agent,0:2].squeeze() for agent in range(self.num_agents)]).squeeze()).tolist()] for n in range(self.H)]).squeeze()
            # timesteps x self.num_agents x self.num_features
            self.mutual_distances_A2F = np.array([[scipy.spatial.distance.cdist(np.array([self.t_A[n,agent,0:2].squeeze() for agent in range(self.num_agents)]).squeeze(),np.array([self.t_F[n,feature,0:2].squeeze() for feature in range(self.num_features)]).squeeze()).tolist()] for n in range(self.H)]).squeeze()

            # timesteps x self.num_agents x self.num_agents
            self.connectivity_matrix_A2A = np.array([(np.array(self.mutual_distances_A2A)<self.comm_distance).squeeze()*1 - np.eye(self.num_agents)]).squeeze()
            # timesteps x self.num_agents x self.num_features
            self.connectivity_matrix_A2F = np.swapaxes(np.swapaxes(raw_dataset['conn_features_oracle'], 0, 2), 1, 2)

            # timesteps x self.num_agents x self.num_agents
            self.x_A2A = np.array([ [[np.abs(np.array(self.mutual_distances_A2A[n][agent]) + self.noise_class(0, self.meas_A2A_std_dist, (self.num_agents,)))] for agent in range(self.num_agents)]
                                    for n in range(self.H)]).squeeze()
            self.x_A2A[self.connectivity_matrix_A2A == 0] = np.nan
            # timesteps x self.num_agents x self.num_features
            self.x_A2F = np.array([ [[np.abs(np.array(self.mutual_distances_A2F[n][agent]) + self.noise_class(0, self.meas_A2F_std_dist, (self.num_features,)))] for agent in range(self.num_agents)]
                                    for n in range(self.H)]).squeeze()
            self.x_A2F[self.connectivity_matrix_A2F == 0] = np.nan
            # timesteps x self.num_agents x num_inputs (4)
            self.x_GNSS = np.array([np.concatenate((self.t_A[n, :, 0:2] + self.noise_class(0, self.meas_GNSS_std_pos, (self.num_agents,2)),
                                     self.t_A[n, :, 2:4] + self.noise_class(0, self.meas_GNSS_std_vel, (self.num_agents,2))), 1) 
                                      for n in range(self.H)]).squeeze()

        # Load dataset
        else:
            torch_dataset = torch.load(self.DB_path_pt)

        if not bool_load_dataset or not os.path.exists(self.DB_path_pt):

            if normalize:
                # Normalize (max-min)
                self.t_A[:,:,0:2] = normalize_numpy(self.t_A[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
                self.t_A[:,:,2:4] = normalize_numpy(self.t_A[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')

                self.t_F[:,:,0:2] = normalize_numpy(self.t_F[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
                self.t_F[:,:,2:4] = normalize_numpy(self.t_F[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')

                self.x_A2A = normalize_numpy(self.x_A2A, 0, np.linalg.norm([np.diff(self.limit_pos1), np.diff(self.limit_pos2)]), normalize = 1, type_='minmax')
                self.x_A2F = normalize_numpy(self.x_A2F, 0, np.linalg.norm([np.diff(self.limit_pos1), np.diff(self.limit_pos2)]), normalize = 1, type_='minmax')
                self.x_GNSS[:,:,0:2] = normalize_numpy(self.x_GNSS[:,:,0:2], np.array([self.limit_pos1[0], self.limit_pos2[0]]), np.array([self.limit_pos1[1], self.limit_pos2[1]]), normalize = 1, type_='minmax')
                self.x_GNSS[:,:,2:4] = normalize_numpy(self.x_GNSS[:,:,2:4], np.array([self.limit_vel1[0], self.limit_vel2[0]]), np.array([self.limit_vel1[1], self.limit_vel2[1]]), normalize = 1, type_='minmax')

            torch_dataset = TensorDataset(torch.tensor(self.t_A).float(), 
                                        torch.tensor(self.t_F).float(), 
                                        torch.tensor(self.x_A2A).float(), 
                                        torch.tensor(self.x_A2F).float(),
                                        torch.tensor(self.x_GNSS).float(), 

                                        torch.tensor(self.mutual_distances_A2A).float(), 
                                        torch.tensor(self.mutual_distances_A2F).float(), 
                                        torch.tensor(self.connectivity_matrix_A2A).float(), 
                                        torch.tensor(self.connectivity_matrix_A2F).float(),  
                                        )   
            numpy_dataset = [np.float32(self.t_A), 
                            np.float32(self.t_F), 
                            np.float32(self.x_A2A), 
                            np.float32(self.x_A2F), 
                            np.float32(self.x_GNSS), 

                            np.float32(self.mutual_distances_A2A), 
                            np.float32(self.mutual_distances_A2F), 
                            np.float32(self.connectivity_matrix_A2A), 
                            np.float32(self.connectivity_matrix_A2F)]

            # Delete variables
            del self.t_A; del self.t_F; del self.x_A2A; del self.x_A2F; del self.x_GNSS;
            del self.mutual_distances_A2A; del self.mutual_distances_A2F; del self.connectivity_matrix_A2A; del self.connectivity_matrix_A2F

        if bool_save_dataset:
            torch.save(torch_dataset, self.DB_path_pt)

        self.dataset_output_names = ['t_A', 't_F', 'x_A2A', 'x_A2F', 'x_GNSS', 
                                     'mutual_distances_A2A', 'mutual_distances_A2F', 'connectivity_matrix_A2A', 'connectivity_matrix_A2F']

        if train:
            loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
            self.params.num_train_batches = len(loader)
            print_name_and_value(self.params.num_train_batches)
        else:
            loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
            self.params.num_valid_batches = len(loader)
            print_name_and_value(self.params.num_valid_batches)
        
        self.params.dataset_output_names = self.dataset_output_names

        self.params.update_all()

        return loader, numpy_dataset, raw_dataset_original


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
        
