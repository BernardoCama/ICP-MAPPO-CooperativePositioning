import matplotlib.pyplot as plt
import os 
import sys
import numpy as np
import os 
import sys
import numpy as np
import importlib
import torch
import platform

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


# Specific experiment
# ARTIFICIAL
exp_name = 'Exp_speed_densities_graph_connection'

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


valid_loader, valid_numpy_dataset, valid_raw_dataset = dataset_class_instance.return_dataset_tracking(train = 0, normalize = 0, shuffle = 0, bool_save_dataset = 0, bool_load_dataset = 0)
valid_dict_dataset = {dataset_output_name:valid_numpy_dataset[index_name] for index_name,(dataset_output_name) in enumerate(dataset_class_instance.dataset_output_names)}



##############################################################################################################################
##############################################################################################################################
### HISTOGRAMS OF VELOCITIES OF TRAINING POINTS IN PYTHON
# Extract the velocities along the x and y axes
# valid_dict_dataset['t_A'].shape = (n_samples, n_vehicles, 4 (posx, posy, velx, vely))
velx = valid_dict_dataset['t_A'][:, :, 2].flatten()
vely = valid_dict_dataset['t_A'][:, :, 3].flatten()
# Convert into km/h
velx = velx * 3.6
vely = vely * 3.6

# Compute the mean and std of velocities along the x and y axes
mean_velx = np.mean(velx)
std_velx = np.std(velx)
mean_vely = np.mean(vely)
std_vely = np.std(vely)
# Compute the absolute velocity for each point and then its mean and std
absolute_velocities = np.sqrt(velx**2 + vely**2)
mean_abs_velocity = np.mean(absolute_velocities)
std_abs_velocity = np.std(absolute_velocities)
min_abs_velocity = np.min(absolute_velocities)
max_abs_velocity = np.max(absolute_velocities)
print(f"Mean velocity along X: {mean_velx:.2f}, Std: {std_velx:.2f}")
print(f"Mean velocity along Y: {mean_vely:.2f}, Std: {std_vely:.2f}")
print(f"Mean absolute velocity: {mean_abs_velocity:.2f}, Std: {std_abs_velocity:.2f}")
print(f"Min absolute velocity: {min_abs_velocity:.2f}, Max: {max_abs_velocity:.2f}")


# TODO PLOT 1 HISTOGRAM OF VELOCITIES ON X AND Y ON THE SAME HISTOGRAM
plt.figure(figsize=(10, 6))
plt.hist(velx, bins=30, alpha=0.5, label='X-axis Velocities', density=True, edgecolor='black')
plt.hist(vely, bins=30, alpha=0.5, label='Y-axis Velocities', density=True, edgecolor='black')

# Adding labels and title
plt.xlabel('Velocity [km/h]', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Density Histogram of Velocities in X and Y Directions', fontsize=16)
plt.legend(fontsize=14)

# Save the plot if needed
file_name = 'Velocity_Histogram'
plt.savefig(os.path.join(Figures_dir, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()


# TODO PLOT 2 HISTOGRAM OF ABSOLUTE VELOCITIES 
plt.figure(figsize=(10, 6))
plt.hist(absolute_velocities, bins=30, alpha=0.5, label='Absolute Velocities', density=True, edgecolor='black')

# Adding labels and title
plt.xlabel('Velocity [km/h]', fontsize=16)
plt.ylabel('Density', fontsize=16)
# plt.title('Density Histogram of Absolute Velocities', fontsize=16)
plt.legend(fontsize=14)

# Save the plot if needed
file_name = 'Velocity_absolute_Histogram'
plt.savefig(os.path.join(Figures_dir, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()



##############################################################################################################################
##############################################################################################################################
### PLOT THE SCATTERPLOT OF THE VEHICLE 0 POSITION OVER TIME (WITH COLOR INDICATING THE VELOCITY)
# Random sample selected vehicles
# num_selected_vehicles = 1
# selected_vehicles = np.random.choice(valid_dict_dataset['t_A'].shape[1], num_selected_vehicles, replace=False)
selected_vehicles = [0]
# Sample the time steps once every step time steps
step = 1
time_steps = np.arange(0, valid_dict_dataset['t_A'].shape[0]//1, step)


# For each selected vehicle, plot the position of vehicle 0 over time
plt.figure(figsize=(10, 10))
for vehicle in selected_vehicles:
    # valid_dict_dataset['t_A'].shape = (n_samples, n_vehicles, 4 (posx, posy, velx, vely))
    # Use the time steps sampled
    posx = valid_dict_dataset['t_A'][time_steps, vehicle, 0]
    posy = valid_dict_dataset['t_A'][time_steps, vehicle, 1]
    velx = valid_dict_dataset['t_A'][time_steps, vehicle, 2]
    vely = valid_dict_dataset['t_A'][time_steps, vehicle, 3]
    # Convert into km/h
    velx = velx * 3.6
    vely = vely * 3.6
    # Compute the absolute velocity for each point
    absolute_velocities = np.sqrt(velx**2 + vely**2)
    # Plot the scatter plot
    # Do not use the line around the points
    plt.scatter(posx, posy, c=absolute_velocities, cmap='jet', s=100, label=f'Vehicle {vehicle+1}', edgecolors='none', alpha=0.5)

# Set the axis limits
plt.xlim([-14, 200])
plt.ylim([100, 313])
# Same axis scale
plt.gca().set_aspect('equal', adjustable='box')

# Adding labels and title
plt.xlabel('Position X [m]', fontsize=16)
plt.ylabel('Position Y [m]', fontsize=16)
plt.title('Position of Vehicle 0 Over Time', fontsize=16)
# Create colorbar
cbar = plt.colorbar(label='Velocity [km/h]')
cbar.set_label('Velocity [km/h]', fontsize=14)  # Set label font size
cbar.ax.tick_params(labelsize=12)  # Set the font size of the colorbar ticks


# Save the plot if needed
file_name = 'Position_Over_Time'
plt.savefig(os.path.join(Figures_dir, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()


##############################################################################################################################
##############################################################################################################################
### PLOT THE NUMBER OF CONNECTIONS OF VEHICLES FOR EACH TIME STEP

# Random sample selected vehicles
num_selected_vehicles = 1
selected_vehicles = np.random.choice(valid_dict_dataset['t_A'].shape[1], num_selected_vehicles, replace=False)

# For each selected vehicle, plot the number of connections for each time step
plt.figure(figsize=(10, 6))
for vehicle in selected_vehicles:
    # valid_dict_dataset['connectivity_matrix_A2A'].shape = (n_samples, n_vehicles, n_vehicles)
    connections = valid_dict_dataset['connectivity_matrix_A2A'][:, vehicle, :].sum(axis=1)
    plt.plot(connections, label=f'Vehicle {vehicle+1}')

# Adding labels and title
plt.xlabel('Time Step', fontsize=16)
plt.ylabel('Number of Connections', fontsize=16)
plt.title('Number of Connections of Vehicles Over Time', fontsize=16)
plt.legend(fontsize=14)

# Save the plot if needed
file_name = 'Connections_Over_Time'
plt.savefig(os.path.join(Figures_dir, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(Figures_dir, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()


