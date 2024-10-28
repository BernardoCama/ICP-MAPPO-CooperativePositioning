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
from Classes.utils.utils import mkdir
from Classes.solver.solver_ICP import Solver_ICP
from Classes.solver.solver_MARL import Solver_MARL 
from Classes.plotting.plotting import plot_carla_MARL_training_results, plot_carla_MARL_ICP_testing_results

# Setting
train = 0
resume_train = 0
plot_train_results = 0
test = 0
plot_test_results = 1


# Specific experiment
# ARTIFICIAL
exp_name = 'test1'

exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(exp_dir, 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results')
Figures_dir = os.path.join(exp_dir, 'Figures')
mkdir(exp_dir)
mkdir(saved_models_dir)
mkdir(output_results_dir)
mkdir(Figures_dir)

# General parameters
parameters_name = exp_name

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
# TRAINING

# Update parameters before training
params.update_all()
if resume_train:
    train_result_MARL = solver_MARL.load_model_and_train_result() 
else:
    train_result_MARL = {}
if train:
    solver_MARL.train(train_result_MARL)

###########################################################################################
###########################################################################################
# LOADING MODEL - TESTING - VISUALIZATION

# cuda for testing
# solver_MARL.set_cuda_device(0)

# Load pretrained model and train results
if plot_train_results:
    train_result_MARL = solver_MARL.load_model_and_train_result() 
    plot_carla_MARL_training_results(train_result_MARL, dataset_class_instance, params, plt_show = 0)

if test:
    train_result_MARL = solver_MARL.load_model_and_train_result() 

    MonteCarlo = 40
    test_results_MARL = {}
    test_results_ICP = {}
    for mc in tqdm(range(MonteCarlo)):

        valid_loader, valid_numpy_dataset, valid_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 0, normalize = 1, shuffle = 0, bool_save_dataset = 0, bool_load_dataset = 0)

        # MARL
        result_MARL = solver_MARL.test(valid_numpy_dataset)

        # ICP
        result_ICP = solver_ICP.test(valid_raw_dataset)

        test_results_MARL[mc] = result_MARL
        test_results_ICP[mc] = result_ICP

        solver_MARL.save_test_result(test_results_MARL) 
        solver_ICP.save_test_result(test_results_ICP) 


test_results_MARL = solver_MARL.load_test_result()
test_results_ICP = solver_ICP.load_test_result()
if plot_test_results:
    plot_carla_MARL_ICP_testing_results(test_results_MARL, test_results_ICP, dataset_class_instance, params, plt_show = 0)

