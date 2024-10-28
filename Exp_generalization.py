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
from Classes.plotting.plotting import plot_carla_MARL_exp_generalization_results

# Setting
test = 0
plot_test_results = 1



###########################################################################################
###########################################################################################
# TOWN2 - MARL


# Specific experiment
# ARTIFICIAL
exp_training_name = 'test1' # where the models from the training are saved
exp_name = 'Exp_generalization'

exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(os.path.join(EXPERIMENTS_DIR, exp_training_name), 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results_Town2')
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
solver_MARL_Town2 = Solver_MARL(params, dataset_class_instance)

###########################################################################################
###########################################################################################
# LOADING MODEL - TESTING - VISUALIZATION

# cuda for testing
# solver_MARL_Town2.set_cuda_device(0)
if test:
    MonteCarlo = 40
    test_results_MARL = nested_dict()

    for num_features in range(0, 73, 6):  # [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        print(f'Num features: {num_features}')
        solver_MARL_Town2 = Solver_MARL(params, dataset_class_instance)
        solver_MARL_Town2.load_model_and_train_result()

        for mc in tqdm(range(MonteCarlo)):
            _, valid_numpy_dataset, _ = dataset_class_instance.return_dataset_tracking(train=0, normalize=1, shuffle=0, bool_save_dataset=0, bool_load_dataset=0,
                                                                                       specific_dataset_name = 'Town02')

            connectivity_matrix_A2F = valid_numpy_dataset[-1]
            x_A2F = valid_numpy_dataset[3]

            # Adjust feature connectivity across all agents at each timestep
            for t in range(connectivity_matrix_A2F.shape[0]):
                # Aggregate feature detection across all agents for each feature
                detected_features = np.sum(connectivity_matrix_A2F[t], axis=0)  # Summing across the agent axis
                detected_feature_indices = np.where(detected_features > 0)[0]

                if len(detected_feature_indices) > num_features:
                    # More features are detected than allowed, select num_features randomly
                    features_to_keep = np.random.choice(detected_feature_indices, num_features, replace=False)

                    # Build a mask to deactivate unselected features
                    mask = np.ones(detected_features.shape, dtype=bool)
                    mask[features_to_keep] = False

                    # Apply the mask to deactivate features
                    connectivity_matrix_A2F[t, :, mask] = 0

            x_A2F[connectivity_matrix_A2F == 0] = np.nan  # Update feature positions to NaN where deactivated
            valid_numpy_dataset[-1] = connectivity_matrix_A2F
            valid_numpy_dataset[3] = x_A2F

            result_MARL = solver_MARL_Town2.test(valid_numpy_dataset)
            test_results_MARL[num_features][mc] = result_MARL

            # UNCOMMENT TO SAVE RESULTS
            solver_MARL_Town2.save_test_result(test_results_MARL)

test_results_MARL_Town2 = solver_MARL_Town2.load_test_result()


# Stop here
# sys.exit()


###########################################################################################
###########################################################################################
# TOWN10 - MARL


# Specific experiment
# ARTIFICIAL
exp_training_name = 'test2' # where the models from the training are saved
exp_name = 'Exp_generalization'

exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(os.path.join(EXPERIMENTS_DIR, exp_training_name), 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results_Town10')
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
solver_MARL_Town10 = Solver_MARL(params, dataset_class_instance)

###########################################################################################
###########################################################################################
# LOADING MODEL - TESTING - VISUALIZATION

# cuda for testing
# solver_MARL_Town10.set_cuda_device(0)
if test:
    MonteCarlo = 40
    test_results_MARL = nested_dict()

    for num_features in range(0, 73, 6):  # [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        print(f'Num features: {num_features}')
        solver_MARL_Town10 = Solver_MARL(params, dataset_class_instance)
        solver_MARL_Town10.load_model_and_train_result()

        for mc in tqdm(range(MonteCarlo)):
            _, valid_numpy_dataset, _ = dataset_class_instance.return_dataset_tracking(train=0, normalize=1, shuffle=0, bool_save_dataset=0, bool_load_dataset=0, 
                                                                                        specific_dataset_name = 'Town10')

            connectivity_matrix_A2F = valid_numpy_dataset[-1]
            x_A2F = valid_numpy_dataset[3]

            # Adjust feature connectivity across all agents at each timestep
            for t in range(connectivity_matrix_A2F.shape[0]):
                # Aggregate feature detection across all agents for each feature
                detected_features = np.sum(connectivity_matrix_A2F[t], axis=0)  # Summing across the agent axis
                detected_feature_indices = np.where(detected_features > 0)[0]

                if len(detected_feature_indices) > num_features:
                    # More features are detected than allowed, select num_features randomly
                    features_to_keep = np.random.choice(detected_feature_indices, num_features, replace=False)

                    # Build a mask to deactivate unselected features
                    mask = np.ones(detected_features.shape, dtype=bool)
                    mask[features_to_keep] = False

                    # Apply the mask to deactivate features
                    connectivity_matrix_A2F[t, :, mask] = 0

            x_A2F[connectivity_matrix_A2F == 0] = np.nan  # Update feature positions to NaN where deactivated
            valid_numpy_dataset[-1] = connectivity_matrix_A2F
            valid_numpy_dataset[3] = x_A2F

            result_MARL = solver_MARL_Town10.test(valid_numpy_dataset)
            test_results_MARL[num_features][mc] = result_MARL

            # UNCOMMENT TO SAVE RESULTS
            solver_MARL_Town10.save_test_result(test_results_MARL)

test_results_MARL_Town10 = solver_MARL_Town10.load_test_result()



if plot_test_results:
    plot_carla_MARL_exp_generalization_results(test_results_MARL_Town2, test_results_MARL_Town10, params, plt_show = 0, 
                                           plot_rmse_vs_num_max_features=1)

