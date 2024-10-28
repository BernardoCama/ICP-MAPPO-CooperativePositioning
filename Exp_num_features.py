import os 
import sys
import numpy as np
import importlib
import torch
import platform
from tqdm import tqdm

###########################################################################################
###########################################################################################
# DEFINITIONS DIRECTORIES - CLASSES - EXPERIMENTS - DATASETS - MODEL - OPTIMIZER - BNN ALGORITHM

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(cwd)[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(cwd))

# Importing Generic Classes
from Classes.utils.utils import mkdir, nested_dict
from Classes.solver.solver_ICP import Solver_ICP
from Classes.solver.solver_MARL import Solver_MARL 
from Classes.plotting.plotting import plot_carla_MARL_exp_num_features_results

# Setting
test = 0
plot_test_results = 1


# Specific experiment
# ARTIFICIAL
exp_training_name = 'test1' # where the models from the training are saved
exp_name = 'Exp_num_features'

exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(os.path.join(EXPERIMENTS_DIR, exp_training_name), 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results')
Figures_dir = os.path.join(exp_dir, 'Figures')
mkdir(exp_dir)
mkdir(saved_models_dir)
mkdir(output_results_dir)
mkdir(Figures_dir)

# General parameters
parameters_name = exp_training_name

parameters_dir = os.path.join(EXPERIMENTS_DIR, parameters_name, 'param.py')
package_params = parameters_dir.split('MARL_Cooperative_Positioning')[1].replace(os.path.sep, '.')[1:-3]
params = getattr(importlib.import_module(package_params), 'PARAMS')()
params({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'EXPERIMENTS_DIR':EXPERIMENTS_DIR})
params({'exp_name':exp_name, 'exp_dir':exp_dir, 'saved_models_dir':saved_models_dir, 'output_results_dir':output_results_dir, 'Figures_dir':Figures_dir})
params({'parameters_name':parameters_name, 'parameters_dir':parameters_dir, 'package_params':package_params})

# Reproducibility
np.random.seed(params.seed)
torch.manual_seed(params.seed)

# OS
OS = platform.system()
params({'OS':OS})

# Dataset
dataset_name = params.dataset_name
DB_name = params.dataset_name

dataset_dir = os.path.join(CLASSES_DIR, 'dataset', dataset_name + '.py')
db_dir = os.path.join(DB_DIR, dataset_name)
package_dataset = dataset_dir.split('MARL_Cooperative_Positioning')[1].replace(os.path.sep, '.')[1:-3]
params({'dataset_dir':dataset_dir, 'DB_name':DB_name, 'db_dir':db_dir, 'package_dataset':package_dataset})
dataset_class_instance = getattr(importlib.import_module(package_dataset), 'DATASET')(params)

# Resources
params({'use_cuda': torch.cuda.is_available()})

###########################################################################################
###########################################################################################
# IMPORTING DATASETS - SOLVER(MODEL)

# Dataset
# train_loader, train_numpy_dataset, train_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 1)
# valid_loader, valid_numpy_dataset, valid_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 0, normalize = 0)
# dataset_class_instance.show_dataset(train_loader)
log_train_step = 1 # period in number of epochs after which we save model parameters and training statistics
log_valid_step = 1 # period in number of epochs after which we save validation statistics
params({'log_train_step':log_train_step, 'log_valid_step':log_valid_step})

# Solvers
solver_ICP = Solver_ICP(params)
solver_MARL = Solver_MARL(params, dataset_class_instance)

###########################################################################################
###########################################################################################
# LOADING MODEL - TESTING - VISUALIZATION

# cuda for testing
# solver_MARL.set_cuda_device(0)


if test:
    

    MonteCarlo = 40
    test_results_MARL = nested_dict()
    # test_results_MARL = solver_MARL.load_test_result()
    # test_results_ICP =  nested_dict()
    # test_results_ICP = solver_ICP.load_test_result()
    for num_features in range(0, 15, 1): # Max number of detected features

        print(f'Num features: {num_features}')

        solver_MARL = Solver_MARL(params, dataset_class_instance)
        solver_MARL.load_model_and_train_result() 

        for mc in tqdm(range(MonteCarlo)):

            valid_loader, valid_numpy_dataset, valid_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 0, normalize = 1, shuffle = 0, bool_save_dataset = 0, bool_load_dataset = 0)

            # Modify connectivity matrix for MARL
            # connectivity_matrix_A2A = valid_numpy_dataset[-2]
            connectivity_matrix_A2F = valid_numpy_dataset[-1]
            # x_A2A = valid_numpy_dataset[2]
            x_A2F = valid_numpy_dataset[3]

            # For each timestep and each agent
            for t in range(connectivity_matrix_A2F.shape[0]):
                for a in range(connectivity_matrix_A2F.shape[1]):
                    # Find the indices where features are detected (value is 1)
                    features_detected_indices = np.where(connectivity_matrix_A2F[t, a] == 1)[0]
                    
                    # If more features are detected than allowed
                    if len(features_detected_indices) > num_features:
                        # Randomly choose indices to keep, up to the allowed number of features
                        indices_to_keep = np.random.choice(features_detected_indices, num_features, replace=False)
                        
                        # Set all other detections to 0
                        indices_to_discard = np.setdiff1d(features_detected_indices, indices_to_keep)
                        connectivity_matrix_A2F[t, a, indices_to_discard] = 0

            # x_A2A[connectivity_matrix_A2A == 0] = np.nan
            x_A2F[connectivity_matrix_A2F == 0] = np.nan
            # valid_numpy_dataset[-2] = connectivity_matrix_A2A
            valid_numpy_dataset[-1] = connectivity_matrix_A2F
            # valid_numpy_dataset[2] = x_A2A
            valid_numpy_dataset[3] = x_A2F

            # Modify connectivity matrix for ICP
            connectivity_matrix_A2F = valid_raw_dataset['conn_features'] # (num_agents, num_features, timesteps)
            connectivity_matrix_A2F = np.swapaxes(np.swapaxes(connectivity_matrix_A2F, 0, 2), 1, 2)

            # For each timestep and each agent
            for t in range(connectivity_matrix_A2F.shape[0]):
                for a in range(connectivity_matrix_A2F.shape[1]):
                    # Find the indices where features are detected (value is 1)
                    features_detected_indices = np.where(connectivity_matrix_A2F[t, a] == 1)[0]
                    
                    # If more features are detected than allowed
                    if len(features_detected_indices) > num_features:
                        # Randomly choose indices to keep, up to the allowed number of features
                        indices_to_keep = np.random.choice(features_detected_indices, num_features, replace=False)
                        
                        # Set all other detections to 0
                        indices_to_discard = np.setdiff1d(features_detected_indices, indices_to_keep)
                        connectivity_matrix_A2F[t, a, indices_to_discard] = 0

            connectivity_matrix_A2F = np.swapaxes(np.swapaxes(connectivity_matrix_A2F, 2, 0), 0, 1)
            valid_raw_dataset['conn_features'] = connectivity_matrix_A2F

            # MARL
            result_MARL = solver_MARL.test(valid_numpy_dataset)

            # ICP
            # result_ICP = solver_ICP.test(valid_raw_dataset)

            test_results_MARL[num_features][mc] = result_MARL
            # test_results_ICP[num_features][mc] = result_ICP

            # UNCOMMENT TO SAVE RESULTS
            solver_MARL.save_test_result(test_results_MARL) 
            # solver_ICP.save_test_result(test_results_ICP) 


test_results_MARL = solver_MARL.load_test_result()
test_results_ICP = solver_ICP.load_test_result()
if plot_test_results:
    plot_carla_MARL_exp_num_features_results(test_results_MARL, test_results_ICP, dataset_class_instance, params, plt_show = 0, 
                                           plot_rmse_vs_num_max_features=1)

