from typing import Any
import math

class PARAMS(object):

    def __init__(self, config = {}):

        self.DEFAULTS = {

                'seed': 2,

                # DATASET GENERAL
                'dataset_name':'Carla', # 'Artificial_2D', 'Carla'
                'H': 1500, # deafult time horizon [s]
                'start_time': 1,
                # # 'size_train_dataset':250,
                # # 'size_valid_dataset':150,
                'batch_size': int(1500/750),  # int(1500/300) Number of environments simulated

                # DATASET SPECIFIC
                'num_agents': 20, # deafult
                'num_features': 146, # deafult # Town10
                'T_between_timestep': 0.2, # [s] Carla 0.2
                'limit_behavior': 'reflection',
                'comm_distance': 1000, # [m]
                'limit':{'Town02': {'limit_pos1': [-14, 200], # [m] 
                                    'limit_pos2': [100, 313], # [m] 
                                    'limit_vel1': [-20, 20], # [m/s] 
                                    'limit_vel2': [-20, 20]}, # [m/s]
                         'Town10': {'limit_pos1': [-120, 120], # [m]   
                                    'limit_pos2': [-74, 150], # [m] 
                                    'limit_vel1': [-20, 20], # [m/s] 
                                    'limit_vel2': [-20, 20]} # [m/s]
                                    },
                'noise_type': 'Gaussian',
                # Trajectories
                'setting_trajectories_A': 'not_defined', # 'not_defined', 'star', 'spiral'
                'setting_trajectories_F': 'not_defined', # 'not_defined', 'star', 'spiral'
                # Real initial conditions
                'initial_A_mean_pos1': 10, # [m] # Not used in Carla
                'initial_A_mean_pos2': 10, # [m] # Not used in Carla
                'initial_A_mean_vel1': 0, # [m/s] # Not used in Carla
                'initial_A_mean_vel2': 0, # [m/s] # Not used in Carla
                'initial_A_mean_acc1': 0, # [m/s^2] # Not used in Carla
                'initial_A_mean_acc2': 0, # [m/s^2] # Not used in Carla
                'initial_F_mean_pos1': -10, # [m] # Not used in Carla
                'initial_F_mean_pos2': -10, # [m] # Not used in Carla
                'initial_F_mean_vel1': 0, # [m/s] # Not used in Carla
                'initial_F_mean_vel2': 0, # [m/s] # Not used in Carla
                'initial_F_mean_acc1': 0, # [m/s^2] # Not used in Carla
                'initial_F_mean_acc2': 0, # [m/s^2] # Not used in Carla
                'initial_A_std_pos1': 5, # [m] # Not used in Carla
                'initial_A_std_pos2': 5, # [m] # Not used in Carla
                'initial_A_std_vel1': 0, # [m/s] # Not used in Carla
                'initial_A_std_vel2': 0, # [m/s] # Not used in Carla
                'initial_A_std_acc1': 0, # [m/s^2] # Not used in Carla
                'initial_A_std_acc2': 0, # [m/s^2] # Not used in Carla
                'initial_F_std_pos1': 5, # [m] # Not used in Carla
                'initial_F_std_pos2': 5, # [m] # Not used in Carla
                'initial_F_std_vel1': 0, # [m/s] # Not used in Carla
                'initial_F_std_vel2': 0, # [m/s] # Not used in Carla
                'initial_F_std_acc1': 0, # [m/s^2] # Not used in Carla
                'initial_F_std_acc2': 0, # [m/s^2] # Not used in Carla
                # Real Motion
                'motion_A_std_pos':1, # [m] # Not used in Carla
                'motion_A_std_vel':0.1, # [m/s] # Not used in Carla
                'motion_A_std_acc':0, # [m/s^2] # Not used in Carla
                'motion_F_std_pos':0.1, # [m] # Not used in Carla
                'motion_F_std_vel':0, # [m/s] # Not used in Carla
                'motion_F_std_acc':0, # [m/s^2] # Not used in Carla
                # Real Measurements
                'meas_A2F_std_dist': 2, # [m]  
                'meas_A2A_std_dist': 2, # [m]  
                'meas_GNSS_std_pos': 2, # [m] 
                'meas_GNSS_std_vel': 0.1, # [m/s] 
                'meas_GNSS_std_acc': 0, # [m/s^2] 

                # DATASET BOOL
                'bool_load_dataset': 0,
                'bool_save_dataset': 1,
                # 'bool_shuffle':1,

                # OPTIMIZER
                'optimizer_name': 'Adam',
                'lr_lstm': 1e-5,          # Learning rate
                'lr_actor': 1e-5,          # Learning rate
                'lr_critic': 1e-5,          # Learning rate
                # 'bool_clip_grad_norm': 0,

                # ALGORITHM
                'ICP_algorithm':'ICP',
                # Motion model
                'motion_model_A_std_pos':0, # [m] 
                'motion_model_A_std_vel':1, # [m/s] 
                'motion_model_A_std_acc':0, # [m/s^2] 
                'motion_model_F_std_pos':0, # [m] 
                'motion_model_F_std_vel':0, # [m/s] 
                'motion_model_F_std_acc':0, # [m/s^2] 
                # Measurement model
                'meas_model_A2F_std_dist': 2, # [m]  
                'meas_model_A2A_std_dist': 2, # [m]  
                'meas_model_GNSS_std_pos': 2, # [m] 
                'meas_model_GNSS_std_vel': 0.1, # [m/s] 
                'meas_model_GNSS_std_acc': 0, # [m/s^2] 
                # Priors
                'prior_A_std_pos': 1, # [m]  
                'prior_A_std_vel': 0.1, # [m/s] 
                'pior_F_std_pos': 100, # [m]  

                # MODELS DEFINITION
                # Lstm
                'LSTM_bidirectional': 1, 
                'LSTM_num_layers': 2,
                'LSTM_hidden': 256, 
                'LSTM_hidden_clip': 1e6, 
                'L': 750, # 300 deafult sequence length (time horizon splitting) [s]

                # Actor
                'actor_max_grad_norm': 10, 
                'actor_epsilon_clip': 1e-6,

                # Critic
                'critic_hidden': 256,


                # MARL HYPER-PARAMETERS
                'alpha_entropy': 10, # 0.1
                'beta_reward': 0.01, # 0.05 [m]
                'gamma_discount': 0.99,
                'epsilon_clip': 0.2,
                'lambda_GAE': -1,


                # TRAINING PARAMETERS
                'num_steps': 2000, #2000, 
                'num_epochs_Adam' : -1,  


                # # TRAINING - STATISTICS BOOL
                # 'bool_validate_model':1,            # Validate the model
                # 'bool_return_train_accuracy_metrics':0,   # Compute accuracy metrics when performing training
                # 'bool_return_valid_loss_metrics':0,       # Compute loss metric when performing validation
                # 'bool_return_valid_accuracy_metrics':1,   # Compute accuracy metrics when performing validation
                # 'bool_plot_dataset':0,              # Plot dataset when created
                # 'bool_pretrained_model': None,      # Load pretrained model when start training
                # 'bool_print_network': 1,            # Print network composition
                # 'bool_save_training_info': 1,       # Save training statistics
                # 'bool_plot_training':1,             # Plot the training statistics during training
        }   

        self.DEFAULTS.update(config)
        self.__dict__.update(self.DEFAULTS, **config)

    def __call__(self, config = {}, *args: Any, **kwds: Any) -> Any:
        self.DEFAULTS.update(config)
        self.__dict__.update(self.DEFAULTS, **config)

    def update_all(self):
        # Update self.DEFAULTS with every self.variable_name
        for key, value in self.__dict__.items():
            if key != 'DEFAULTS':
                self.DEFAULTS[key] = value

        # Update self with all the variables in self.DEFAULTS
        self.__dict__.update(self.DEFAULTS)