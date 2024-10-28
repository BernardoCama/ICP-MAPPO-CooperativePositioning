import os 
import sys
import numpy as np
import importlib
import torch
import platform
import copy

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
from Classes.utils.utils import mkdir, nested_dict, make_symmetric
from Classes.solver.solver_ICP import Solver_ICP
from Classes.solver.solver_MARL import Solver_MARL 
from Classes.plotting.plotting import plot_carla_MARL_exp_timesteps_A2A_connections

# Setting
test = 0
plot_test_results = 1


# Specific experiment
# ARTIFICIAL
exp_training_name = 'test1' # where the models from the training are saved
exp_name = 'Exp_num_agents'

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
valid_loader, valid_numpy_dataset, valid_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 0, normalize = 0)
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




test_results_ICP =  nested_dict()
for num_agents in range(0, params.num_agents):

    # Modify connectivity matrix for MARL
    connectivity_matrix_A2A = copy.deepcopy(valid_numpy_dataset[-2])
    connectivity_matrix_A2F = copy.deepcopy(valid_numpy_dataset[-1])

    dim0, dim1, dim2 = params.H, params.num_agents, params.num_agents
    indices_MARL_A2A = np.array([np.random.choice(dim2, params.num_agents - num_agents, replace=False) for _ in range(dim0 * dim1)]).reshape(dim0, dim1, params.num_agents - num_agents)
    n_range_MARL_A2A = np.arange(dim0)[:, None, None]
    a_range_MARL_A2A = np.arange(dim1)[None, :, None]
    connectivity_matrix_A2A[n_range_MARL_A2A, a_range_MARL_A2A, indices_MARL_A2A] = 0

    test_results_ICP[num_agents] = {'connectivity_matrix_A2A':connectivity_matrix_A2A, 
                                    'connectivity_matrix_A2F':connectivity_matrix_A2F}


test_results_MARL = solver_MARL.load_test_result()
if plot_test_results:
    plot_carla_MARL_exp_timesteps_A2A_connections(test_results_MARL, test_results_ICP, dataset_class_instance, params, plt_show = 0)

