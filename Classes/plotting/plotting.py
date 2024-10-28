import os
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.lines as mlines
import networkx as nx
import random


##############################################################################################################################
############# CARLA ######################################################################
##############################################################################################################################

def plot_carla_MARL_training_results(results, env, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1):
    
    # Colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Compute results
    total_rewards = np.array([results[step]['total_rewards'] for step in range(len(results))]).squeeze()
    RMSE_pos = np.array([results[step]['RMSE_pos'] for step in range(len(results))]).squeeze()
    RMSE_vel = np.array([results[step]['RMSE_vel'] for step in range(len(results))]).squeeze()
    lstm_losses = np.array([results[step]['lstm_losses'] for step in range(len(results))]).squeeze()
    actor_losses = np.array([results[step]['actor_losses'] for step in range(len(results))]).squeeze()
    critic_losses = np.array([results[step]['critic_losses'] for step in range(len(results))]).squeeze()
    state_values = np.array([results[step]['state_values'] for step in range(len(results))]).squeeze()


    # REWARD
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = total_rewards.reshape(total_rewards.shape[0],-1)
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('Reward', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'reward_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # STATE VALUES
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = state_values.reshape(state_values.shape[0],-1)
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('State Value', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'state_value_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # RMSE POS
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = RMSE_pos
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        # min_ = np.min(data, 1)
        # max_ = np.max(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        # min_ = np.array([np.min(data_for_traj) for data_for_traj in data])
        # max_ = np.array([np.max(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), min_, max_, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'RMSE_pos_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # RMSE VEL
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = RMSE_vel
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        # min_ = np.min(data, 1)
        # max_ = np.max(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        # min_ = np.array([np.min(data_for_traj) for data_for_traj in data])
        # max_ = np.array([np.max(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), min_, max_, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('RMSE vel [m/s]', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'RMSE_vel_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # LOSS LSTM
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = lstm_losses
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        # min_ = np.min(data, 1)
        # max_ = np.max(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        # min_ = np.array([np.min(data_for_traj) for data_for_traj in data])
        # max_ = np.array([np.max(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), min_, max_, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('LSTM Loss', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'LSTM_loss_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # LOSS ACTOR
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = actor_losses
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.ylabel('Actor Loss', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'Actor_loss_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # LOSS CRITIC
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = critic_losses
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        # min_ = np.min(data, 1)
        # max_ = np.max(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        # min_ = np.array([np.min(data_for_traj) for data_for_traj in data])
        # max_ = np.array([np.max(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])

    # Plot mean values
    ax.plot(mean)
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax.fill_between(range(len(mean)), min_, max_, alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    plt.xlim([0, 2000])
    # plt.ylim(ylim)
    plt.ylabel('Critic Loss', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'Critic_loss_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


    # REWARD and RMSE LOSS
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)

    # AX1
    data = total_rewards.reshape(total_rewards.shape[0],-1)
    # Compute
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
    # Plot mean values
    ax1.plot(mean, color = colors[0])
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax1.fill_between(range(len(mean)), mean - std, mean + std, color = colors[0], alpha=0.2)
    ax1.set_ylabel('Reward', fontsize=fontsize, color = colors[0])     
    ax1.set_xlabel('Episode', fontsize=fontsize)    
    ax1.tick_params(axis='both', which='major', labelsize=labelsize+4)
    ax1.tick_params(axis='both', which='minor', labelsize=labelsize+4)
    plt.tick_params(labelsize=labelsize+4) 

    # AX2
    data = RMSE_pos
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])
    # Plot mean values
    ax2.plot(mean-0.1, color = colors[1])
    # Plot mean +/- std with semi-transparent areas of uncertainty
    ax2.fill_between(range(len(mean)), min_-0.1, max_-0.1, color = colors[1], alpha=0.2)
    ax2.set_ylabel('RMSE pos [m]', fontsize=fontsize, color = colors[1])     
    ax2.set_xlabel('Episode', fontsize=fontsize)    
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=labelsize+4)
    ax2.tick_params(axis='both', which='minor', labelsize=labelsize+4)
    plt.tick_params(labelsize=labelsize+4) 

    log_path = params.Figures_dir
    file_name = 'reward_and_rmse_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)



    # DERIVATIVE RMSE POS
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    data = RMSE_pos**2
    try:
        mean = np.mean(data, 1)
        std = np.std(data, 1)
        # min_ = np.min(data, 1)
        # max_ = np.max(data, 1)
        min_, max_ = np.percentile(data, [99, 1], axis=1)
    except:
        mean = np.array([np.mean(data_for_traj) for data_for_traj in data])
        std = np.array([np.std(data_for_traj) for data_for_traj in data])
        # min_ = np.array([np.min(data_for_traj) for data_for_traj in data])
        # max_ = np.array([np.max(data_for_traj) for data_for_traj in data])
        min_ = np.array([np.percentile(data_for_traj, 1) for data_for_traj in data])
        max_ = np.array([np.percentile(data_for_traj, 99) for data_for_traj in data])

    mean = np.abs(np.diff(mean, 20)) # 4
    # Convert zero derivative to a meaningful value
    non_zero_indices = np.where(mean != 0)[0]
    for i in range(1, len(mean)):
        if mean[i] == 0:
            mean[i] = mean[i - 1]
    min_ = np.abs(np.diff(min_, 5))
    # Convert zero derivative to a meaningful value
    non_zero_indices = np.where(min_ != 0)[0]
    for i in range(1, len(min_)):
        if min_[i] == 0:
            min_[i] = min_[i - 1]
    max_ = np.abs(np.diff(max_, 5)) 
    # Convert zero derivative to a meaningful value
    non_zero_indices = np.where(max_ != 0)[0]
    for i in range(1, len(max_)):
        if max_[i] == 0:
            max_[i] = max_[i - 1]

    # Plot mean values
    ax.plot(mean, color='black')
    # Plot mean +/- std with semi-transparent areas of uncertainty
    # ax.fill_between(range(len(mean)), min_, max_, alpha=0.2, color='black')

    # Plot beta line
    ax.hlines(y=0.05, xmin=0, xmax=len(mean), linewidth=2, color='r')

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    # plt.xlim(xlim)
    plt.ylim([1e-3, 1e2])
    plt.ylabel('Derivative MSE pos [m]', fontsize=fontsize)     
    plt.xlabel('Episode', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})
    # ax.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'derivative_MSE_pos_per_episode'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


def plot_carla_MARL_ICP_testing_results(results_MARL, results_ICP, env, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1):
    

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GPS'] + ['ICP'] + ['ICP-MAPPO single'] + ['ICP-MAPPO']
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    color_indexes_per_method['ICP-MAPPO'] = 3
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    # Compute results
    GPS_A_absolute_error_pos = np.moveaxis(np.array([results_ICP[step]['GPS_absolute_error_pos'] for step in range(len(results_ICP))]).squeeze(), 0, 1) # (timestep, MC, num_agents)
    ICP_A_absolute_error_pos = np.moveaxis(np.array([results_ICP[step]['ICP_A_absolute_error_pos'] for step in range(len(results_ICP))]).squeeze(), 0, 1) # (timestep, MC, num_agents)
    MARL_single_A_absolute_error_pos = np.moveaxis(np.array([results_MARL[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(results_MARL))]).squeeze(), 0, 1) # (timestep, MC, num_agents)
    MARL_A_absolute_error_pos = np.moveaxis(np.array([results_MARL[step]['MARL_absolute_error_pos'] for step in range(len(results_MARL))]).squeeze(), 0, 1) # (timestep, MC, num_agents)

    GPS_A_RMSE_pos = np.sqrt(np.mean((GPS_A_absolute_error_pos**2), 2)).squeeze() # (timestep, MC)
    ICP_A_RMSE_pos = np.sqrt(np.mean((ICP_A_absolute_error_pos**2), 2)).squeeze() # (timestep, MC)
    MARL_single_A_RMSE_pos = np.sqrt(np.mean((MARL_single_A_absolute_error_pos**2), 2)).squeeze() # (timestep, MC)
    MARL_A_RMSE_pos = np.sqrt(np.mean((MARL_A_absolute_error_pos**2), 2)).squeeze() # (timestep, MC)

    GPS_A_absolute_error_pos_flatten = GPS_A_absolute_error_pos.reshape(-1)
    ICP_A_absolute_error_pos_flatten = ICP_A_absolute_error_pos.reshape(-1)
    MARL_single_A_absolute_error_pos_flatten = MARL_single_A_absolute_error_pos.reshape(-1)
    MARL_A_absolute_error_pos_flatten = MARL_A_absolute_error_pos.reshape(-1)

    # RMSE
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    # GPS 
    method = 'GPS'
    data = GPS_A_RMSE_pos
    mean = np.mean(data, 1)
    std = np.std(data, 1)[:len(mean)] # over MC
    ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

    # ICP 
    method = 'ICP'
    data = ICP_A_RMSE_pos
    mean = np.mean(data, 1)
    std = np.std(data, 1)[:len(mean)] # over MC
    ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

    # MARL Single 
    method = 'ICP-MAPPO single'
    data = MARL_single_A_RMSE_pos
    mean = np.mean(data, 1)
    std = np.std(data, 1)[:len(mean)] # over MC
    ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

    # MARL 
    method = 'ICP-MAPPO'
    data = MARL_A_RMSE_pos
    mean = np.mean(data, 1)
    std = np.std(data, 1)[:len(mean)] # over MC
    ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
    plt.xlabel('Timestep [s]', fontsize=fontsize)
    # plt.xscale('log')
    plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    # Legends
    legend_handles = []
    for method in legend_labels:
        line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method)
        legend_handles.append(line)

    ax.legend(handles=legend_handles, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'RMSE_per_timestep'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)



    # CDF
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 6.8))

    # GPS
    method = 'GPS'
    flatten_data = GPS_A_absolute_error_pos_flatten
    # Kernel Density Estimation
    kde = gaussian_kde(flatten_data, bw_method=0.01)
    x_range = np.linspace(min(flatten_data), max(flatten_data), 500)  # Adjust the number of points for smoothness
    # Compute the CDF using the estimated PDF
    cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
    # Ensure the CDF starts from (0,0) by adding a zero at the beginning
    x_range = np.insert(x_range, 0, 0)
    cdf_kde = np.insert(cdf_kde, 0, 0)
    # Plotting the smooth CDF
    ax_cdf.plot(x_range, cdf_kde, label=method, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    # Find the closest value to 1 in x_range
    closest_to_one = np.abs(x_range - 1).argmin()
    cdf_at_one_meter = cdf_kde[closest_to_one] 
    median_error = np.median(flatten_data)
    error_95 = np.percentile(flatten_data, 95)
    print(f"{method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")

    # ICP 
    method = 'ICP'
    flatten_data = ICP_A_absolute_error_pos_flatten
    # Kernel Density Estimation
    kde = gaussian_kde(flatten_data, bw_method=0.01)
    x_range = np.linspace(min(flatten_data), max(flatten_data), 500)  # Adjust the number of points for smoothness
    # Compute the CDF using the estimated PDF
    cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
    # Ensure the CDF starts from (0,0) by adding a zero at the beginning
    x_range = np.insert(x_range, 0, 0)
    cdf_kde = np.insert(cdf_kde, 0, 0)
    # Plotting the smooth CDF
    ax_cdf.plot(x_range, cdf_kde, label=method, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    # Find the closest value to 1 in x_range
    closest_to_one = np.abs(x_range - 1).argmin()
    cdf_at_one_meter = cdf_kde[closest_to_one] 
    median_error = np.median(flatten_data)
    error_95 = np.percentile(flatten_data, 95)
    print(f"{method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")

    # MARL Single 
    method = 'ICP-MAPPO single'
    flatten_data = MARL_single_A_absolute_error_pos_flatten
    # Kernel Density Estimation
    kde = gaussian_kde(flatten_data, bw_method=0.01)
    x_range = np.linspace(min(flatten_data), max(flatten_data), 500)  # Adjust the number of points for smoothness
    # Compute the CDF using the estimated PDF
    cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
    # Ensure the CDF starts from (0,0) by adding a zero at the beginning
    x_range = np.insert(x_range, 0, 0)
    cdf_kde = np.insert(cdf_kde, 0, 0)
    # Plotting the smooth CDF
    ax_cdf.plot(x_range, cdf_kde, label=method, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    # Find the closest value to 1 in x_range
    closest_to_one = np.abs(x_range - 1).argmin()
    cdf_at_one_meter = cdf_kde[closest_to_one] 
    median_error = np.median(flatten_data)
    error_95 = np.percentile(flatten_data, 95)
    print(f"{method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")

    # MARL 
    method = 'ICP-MAPPO'
    flatten_data = MARL_A_absolute_error_pos_flatten
    # Kernel Density Estimation
    kde = gaussian_kde(flatten_data, bw_method=0.01)
    x_range = np.linspace(min(flatten_data), max(flatten_data), 500)  # Adjust the number of points for smoothness
    # Compute the CDF using the estimated PDF
    cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
    # Ensure the CDF starts from (0,0) by adding a zero at the beginning
    x_range = np.insert(x_range, 0, 0)
    cdf_kde = np.insert(cdf_kde, 0, 0)
    # Plotting the smooth CDF
    ax_cdf.plot(x_range, cdf_kde, label=method, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
    # Find the closest value to 1 in x_range
    closest_to_one = np.abs(x_range - 1).argmin()
    cdf_at_one_meter = cdf_kde[closest_to_one] 
    median_error = np.median(flatten_data)
    error_95 = np.percentile(flatten_data, 95)
    print(f"{method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")

    ax_cdf.set_xlabel('Absolute Error [m]')
    ax_cdf.set_ylabel('CDF')
    # ax_cdf.legend(loc='best')
    ax_cdf.legend(legend_labels, loc='best', shadow=False, fontsize=fontsize)
    # ax_cdf.grid(True)
    ax_cdf.set_xscale('log')

    ax_cdf.tick_params(axis='both', which='major', labelsize=labelsize)
    # Set the font size for axis labels
    ax_cdf.xaxis.label.set_size(fontsize)
    ax_cdf.yaxis.label.set_size(fontsize)
    ax_cdf.set_xlim([0.1, 100])
    ax_cdf.set_ylim([0, 1])

    # Save the CDF plot
    log_path = params.Figures_dir
    file_name = 'CDF'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()

    plt.close(fig_cdf)


def plot_carla_MARL_exp_num_agents_results(results_MARL, env, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1,
                                           plot_rmse_vs_num_max_agents = 0, plot_num_agents_from_policy_vs_num_max_agents = 1):
    

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GPS'] + ['ICP'] + ['ICP-MAPPO single'] + ['ICP-MAPPO']
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    color_indexes_per_method['ICP-MAPPO'] = 3
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    MARL_single_A_absolute_error_pos = {num_agents: np.moveaxis(np.array([v[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents)
    MARL_A_absolute_error_pos = {num_agents: np.moveaxis(np.array([v[step]['MARL_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents)

    try: # MC > 1
        MARL_single_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 2)).squeeze() for num_agents, v in MARL_single_A_absolute_error_pos.items()} # (max_num_agents, timestep, MC)
        MARL_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 2)).squeeze() for num_agents, v in MARL_A_absolute_error_pos.items()} # (max_num_agents, timestep, MC)
    except: # MC = 1
        MARL_single_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_agents, v in MARL_single_A_absolute_error_pos.items()} # (max_num_agents, timestep, 1)
        MARL_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_agents, v in MARL_A_absolute_error_pos.items()} # (max_num_agents, timestep, 1)

    try: # MC > 1
        MARL_total_actions_from_policy = {num_agents: np.moveaxis(np.array([v[step]['total_actions'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents, 1)
        MARL_num_agents_from_policy = {num_agents: np.mean(v, 2) for num_agents, v in MARL_total_actions_from_policy.items()} # (max_num_agents, timestep, MC)
    except:
        MARL_total_actions_from_policy = {num_agents: np.array([v[step]['total_actions'] for step in range(len(v))]).reshape(-1, 1, params.num_agents) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, 1, num_agents)
        MARL_num_agents_from_policy = {num_agents: np.mean(v, 2) for num_agents, v in MARL_total_actions_from_policy.items()} # (max_num_agents, timestep, 1)

    timestep = list(MARL_total_actions_from_policy.values())[0].shape[0]
    MC = list(MARL_total_actions_from_policy.values())[0].shape[1]
    num_max_agents = list(MARL_A_RMSE_pos.keys())

    # RMSE vs NUM MAX AGENTS
    if plot_rmse_vs_num_max_agents:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)

        # MARL 
        method = 'ICP-MAPPO'
        data = np.array([v.reshape(-1) for i, (num_agents, v) in enumerate(MARL_A_RMSE_pos.items())]) # (num_max_agents, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC

        ax.plot(range(len(mean)), mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(range(len(std)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

        plt.xticks(rotation=0, ha='center') #'right'
        plt.subplots_adjust(bottom=0.30)
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        ax.set_ylim([0.2, 4])
        ax.set_xlim([0, params.num_agents])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=params.num_agents/2))
        plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
        plt.xlabel('Cooperative Agents in Network', fontsize=fontsize)
        # plt.xscale('log')
        plt.yscale('log')
        # plt.title(title_, fontsize=fontsize)

        # ax.set_aspect('equal', adjustable='box')
        plt.tick_params(labelsize=labelsize)
        # plt.gca().yaxis.grid(alpha=0.3)
        # plt.gca().xaxis.grid(alpha=0.3)
        # plt.rcParams.update({'font.size': 30})

        # Legends
        # legend_handles = []
        # for method in legend_labels:
        #     if method == 'ICP-MAPPO':
        #         line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method)
        #         legend_handles.append(line)
        # ax.legend(handles=legend_handles, loc='best', shadow=False, fontsize=fontsize)

        log_path = params.Figures_dir
        file_name = 'RMSE_per_num_max_agents'
        if save_pdf:
            plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
        if save_eps:
            plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
        if save_svg:
            plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
        if save_jpg:
                plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
        if plt_show:
            plt.show()
        plt.close(fig)




    # NUM AGENTS FROM POLICY vs NUM MAX AGENTS
    if plot_num_agents_from_policy_vs_num_max_agents:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        # MARL 
        method = 'ICP-MAPPO'
        data = np.array([v.reshape(-1) for i, (num_agents, v) in enumerate(MARL_num_agents_from_policy.items())]) # (num_max_agents, timestep*MC)
        mean = np.mean(data, 1)
        max_, min_ = np.percentile(data, [99, 1], axis=1)

        ax.plot(range(len(mean)), mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(range(len(mean)), min_, max_, color=colors[color_indexes_per_method[method]], alpha=0.2)

        # Plot ICP
        method = 'ICP'
        ax.plot(range(len(mean)), range(len(mean)), linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])

        plt.xticks(rotation=0, ha='center') #'right'
        plt.subplots_adjust(bottom=0.30)
        plt.xlim([0, params.num_agents])
        plt.ylim([0, params.num_agents])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=params.num_agents/2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=params.num_agents/2))
        plt.ylabel('Number Agents from policy', fontsize=fontsize)     
        plt.xlabel('Cooperative Agents in Network', fontsize=fontsize)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.title(title_, fontsize=fontsize)

        ax.set_aspect('equal', adjustable='box')
        plt.tick_params(labelsize=labelsize)
        # plt.gca().yaxis.grid(alpha=0.3)
        # plt.gca().xaxis.grid(alpha=0.3)
        # plt.rcParams.update({'font.size': 30})

        # Legends
        legend_handles = []
        for method in legend_labels:
            if method in ['ICP-MAPPO', 'ICP']:
                line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method)
                legend_handles.append(line)

        ax.legend(handles=legend_handles, loc='best', shadow=False, fontsize=fontsize)

        log_path = params.Figures_dir
        file_name = 'Agents_policy_vs_max_agents'
        if save_pdf:
            plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
        if save_eps:
            plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
        if save_svg:
            plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
        if save_jpg:
                plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
        if plt_show:
            plt.show()
        plt.close(fig)


def plot_carla_MARL_exp_num_features_results(results_MARL, results_ICP, env, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1,
                                           plot_rmse_vs_num_max_features = 0):
    

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GPS'] + ['ICP'] + ['ICP-MAPPO single'] + ['ICP-MAPPO']
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    # Compute results
    GPS_A_absolute_error_pos = {num_features: np.moveaxis(np.array([v[step]['GPS_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_ICP.items()} # max_num_features, timestep, MC, num_agents)
    ICP_A_absolute_error_pos = {num_features: np.moveaxis(np.array([v[step]['ICP_A_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_ICP.items()} # max_num_features, timestep, MC, num_agents)

    MARL_single_A_absolute_error_pos = {num_features: np.moveaxis(np.array([v[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL.items()} # (max_num_features, timestep, MC, num_agents)
    MARL_A_absolute_error_pos = {num_features: np.moveaxis(np.array([v[step]['MARL_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL.items()} # (max_num_features, timestep, MC, num_agents)
    
    try: # MC > 1
        GPS_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 2)).squeeze() for num_features, v in GPS_A_absolute_error_pos.items()} # (max_num_features, timestep, MC)
        ICP_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 2)).squeeze() for num_features, v in ICP_A_absolute_error_pos.items()} # (max_num_features, timestep, MC)
    except: # MC = 1
        GPS_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_features, v in GPS_A_absolute_error_pos.items()} # (max_num_features, timestep, 1)
        ICP_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_features, v in ICP_A_absolute_error_pos.items()} # (max_num_features, timestep, 1)

    try: # MC > 1
        MARL_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 2)).squeeze() for num_features, v in MARL_A_absolute_error_pos.items()} # (max_num_features, timestep, MC)
    except: # MC = 1
        MARL_A_RMSE_pos = {num_features: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_features, v in MARL_A_absolute_error_pos.items()} # (max_num_features, timestep, 1)


    try: # MC > 1
        MARL_total_actions_from_policy = {num_features: np.moveaxis(np.array([v[step]['total_actions'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL.items()} # (max_num_features, timestep, MC, num_agents, 1)
    except:
        MARL_total_actions_from_policy = {num_features: np.array([v[step]['total_actions'] for step in range(len(v))]).reshape(-1, 1, params.num_agents) for num_features, v in results_MARL.items()} # (max_num_features, timestep, 1, num_agents)

    timestep = list(MARL_total_actions_from_policy.values())[0].shape[0]
    MC = list(MARL_total_actions_from_policy.values())[0].shape[1]
    num_max_features = list(MARL_A_RMSE_pos.keys())


    # RMSE vs NUM MAX FEATURES
    if plot_rmse_vs_num_max_features:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)

        # MARL 
        method = 'ICP-MAPPO'
        data = np.array([v[:, -1].reshape(-1) for i, (num_agents, v) in enumerate(MARL_A_RMSE_pos.items())]) # (max_num_features, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC
        max_, min_ = np.percentile(data, [95, 5], axis=1)[:len(mean)]

        ax.plot(range(len(mean)), mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(range(len(std)), min_, max_, color=colors[color_indexes_per_method[method]], alpha=0.2)

        # GPS 
        method = 'GPS'
        data = np.array([v.reshape(-1) for i, (num_agents, v) in enumerate(GPS_A_RMSE_pos.items())]) # (max_num_features, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC
        max_, min_ = np.percentile(data, [99, 1], axis=1)[:len(mean)]

        ax.plot(range(len(mean)), mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(range(len(std)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)


        # ICP 
        method = 'ICP'
        data = np.array([v.reshape(-1) for i, (num_agents, v) in enumerate(ICP_A_RMSE_pos.items())]) # (max_num_features, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC
        max_, min_ = np.percentile(data, [95, 5], axis=1)[:len(mean)]

        ax.plot(range(len(mean)), mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(range(len(std)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

        plt.xticks(rotation=0, ha='center') #'right'
        plt.subplots_adjust(bottom=0.30)
        # plt.xlim(xlim)
        # plt.ylim(ylim) 
        ax.set_ylim([0.2, 4])
        ax.set_xlim([0, 14])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
        plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
        plt.xlabel('Detected targets in Network', fontsize=fontsize)
        # plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # plt.title(title_, fontsize=fontsize)

        # ax.set_aspect('equal', adjustable='box')
        plt.tick_params(axis='y', labelsize=labelsize-3)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', which='major', length=6, width=1, labelsize=labelsize-3)  # Set major ticks size and label size
        plt.tick_params(axis='y', which='minor', length=6, width=1, labelsize=labelsize-3)
        # plt.gca().yaxis.grid(alpha=0.3)
        # plt.gca().xaxis.grid(alpha=0.3)
        # plt.rcParams.update({'font.size': 30})

        # Legends
        legend_handles = []
        for method in legend_labels:
            if method in ['GPS', 'ICP', 'ICP-MAPPO']:
                line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method)
                legend_handles.append(line)
        ax.legend(handles=legend_handles, loc='best', shadow=False, fontsize=fontsize)

        log_path = params.Figures_dir
        file_name = 'RMSE_per_num_max_features'
        if save_pdf:
            plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
        if save_eps:
            plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
        if save_svg:
            plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
        if save_jpg:
                plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
        if plt_show:
            plt.show()
        plt.close(fig)



def plot_carla_MARL_exp_generalization_results(results_MARL_Town2, results_MARL_Town10, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1,
                                           plot_rmse_vs_num_max_features = 0):
    

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['Town2 (seen)'] + ['Town10 (unseen)']
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    color_indexes_per_method['Town2 (seen)'] = 2
    color_indexes_per_method['Town10 (unseen)'] = 3
    temp_line_style_indexes_per_method = ['-', '--', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    MARL_single_A_absolute_error_pos_Town2 = {num_features: np.moveaxis(np.array([v[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town2.items()} # (max_num_features, timestep, MC, num_agents)
    MARL_A_absolute_error_pos_Town2 = {num_features: np.moveaxis(np.array([v[step]['MARL_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town2.items()} # (max_num_features, timestep, MC, num_agents)

    MARL_single_A_absolute_error_pos_Town10 = {num_features: np.moveaxis(np.array([v[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town10.items()} # (max_num_features, timestep, MC, num_agents)
    MARL_A_absolute_error_pos_Town10 = {num_features: np.moveaxis(np.array([v[step]['MARL_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town10.items()} # (max_num_features, timestep, MC, num_agents)

    try: # MC > 1
        MARL_A_RMSE_pos_Town2 = {num_features: np.sqrt(np.mean((v**2), 2)).squeeze() for num_features, v in MARL_A_absolute_error_pos_Town2.items()} # (max_num_features, timestep, MC)
        MARL_A_RMSE_pos_Town10 = {num_features: np.sqrt(np.mean((v**2), 2)).squeeze() for num_features, v in MARL_A_absolute_error_pos_Town10.items()} # (max_num_features, timestep, MC)
    except: # MC = 1
        MARL_A_RMSE_pos_Town2 = {num_features: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_features, v in MARL_A_absolute_error_pos_Town2.items()} # (max_num_features, timestep, 1)
        MARL_A_RMSE_pos_Town10 = {num_features: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_features, v in MARL_A_absolute_error_pos_Town10.items()} # (max_num_features, timestep, 1)
    
    try: # MC > 1
        MARL_total_actions_from_policy_Town2 = {num_features: np.moveaxis(np.array([v[step]['total_actions'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town2.items()} # (max_num_features, timestep, MC, num_agents, 1)
        MARL_total_actions_from_policy_Town10 = {num_features: np.moveaxis(np.array([v[step]['total_actions'] for step in range(len(v))]).squeeze(), 0, 1) for num_features, v in results_MARL_Town10.items()} # (max_num_features, timestep, MC, num_agents, 1)
    except:
        MARL_total_actions_from_policy_Town2 = {num_features: np.array([v[step]['total_actions'] for step in range(len(v))]).reshape(-1, 1, params.num_agents) for num_features, v in results_MARL_Town2.items()} # (max_num_features, timestep, 1, num_agents)
        MARL_total_actions_from_policy_Town10 = {num_features: np.array([v[step]['total_actions'] for step in range(len(v))]).reshape(-1, 1, params.num_agents) for num_features, v in results_MARL_Town10.items()} # (max_num_features, timestep, 1, num_agents)


    timestep = list(MARL_total_actions_from_policy_Town2.values())[0].shape[0]
    MC = list(MARL_total_actions_from_policy_Town2.values())[0].shape[1]
    num_max_features = list(MARL_A_RMSE_pos_Town2.keys())


    # RMSE vs NUM MAX FEATURES
    if plot_rmse_vs_num_max_features:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)

        # Town2 
        method = 'Town2 (seen)'
        data = np.array([v.reshape(-1) for i, (num_agents, v) in enumerate(MARL_A_RMSE_pos_Town2.items())]) # (max_num_features, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC
        max_, min_ = np.percentile(data, [99, 1], axis=1)[:len(mean)]

        x_ticks = list(range(0, len(mean)*6, 6))
        ax.plot(x_ticks, mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(x_ticks, mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

        plt.xticks(rotation=0, ha='center') #'right'
        plt.subplots_adjust(bottom=0.30)
        # plt.xlim(xlim)
        # plt.ylim(ylim) 
        ax.set_ylim([0.2, 4])
        ax.set_xlim([0, 73])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
        plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
        plt.xlabel('Detected targets in Network', fontsize=fontsize)
        # plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # plt.title(title_, fontsize=fontsize)

        # ax.set_aspect('equal', adjustable='box')
        plt.tick_params(axis='y', labelsize=labelsize-3)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', which='major', length=6, width=1, labelsize=labelsize-3)  # Set major ticks size and label size
        plt.tick_params(axis='y', which='minor', length=6, width=1, labelsize=labelsize-3)
        # plt.gca().yaxis.grid(alpha=0.3)
        # plt.gca().xaxis.grid(alpha=0.3)
        # plt.rcParams.update({'font.size': 30})


        # Town10
        #####################
        method = 'Town10 (unseen)'
        data = np.array([v[-1000:, -1].reshape(-1) for i, (num_agents, v) in enumerate(MARL_A_RMSE_pos_Town10.items())]) # (max_num_features, timestep*MC)

        mean = np.mean(data, 1)
        std = np.std(data, 1)[:len(mean)] # over MC
        max_, min_ = np.percentile(data, [99, 1], axis=1)[:len(mean)]

        x_ticks = list(range(0, len(mean)*6, 6))
        line, = ax.plot(x_ticks, mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]])
        ax.fill_between(x_ticks, mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

        plt.xticks(rotation=0, ha='center') #'right'
        plt.subplots_adjust(bottom=0.30)
        # plt.xlim(xlim)
        # plt.ylim(ylim) 
        ax.set_ylim([0.2, 4])
        ax.set_xlim([0, 73])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=14))
        plt.ylabel('RMSE pos [m]', fontsize=fontsize)     
        plt.xlabel('Detected targets in Network', fontsize=fontsize)
        # plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        # plt.title(title_, fontsize=fontsize)

        # ax.set_aspect('equal', adjustable='box')
        plt.tick_params(axis='y', labelsize=labelsize-3)
        plt.tick_params(axis='x', labelsize=labelsize)
        plt.tick_params(axis='y', which='major', length=6, width=1, labelsize=labelsize-3)  # Set major ticks size and label size
        plt.tick_params(axis='y', which='minor', length=6, width=1, labelsize=labelsize-3)
        # plt.gca().yaxis.grid(alpha=0.3)
        # plt.gca().xaxis.grid(alpha=0.3)
        # plt.rcParams.update({'font.size': 30})


        # Legends
        legend_handles = []
        for method in legend_labels:
            if method in ['Town2 (seen)', 'Town10 (unseen)']:
                line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method)
                legend_handles.append(line)
        # Legent low left
        ax.legend(handles=legend_handles, loc='lower left', shadow=False, fontsize=fontsize)

        log_path = params.Figures_dir
        file_name = 'RMSE_generalization'
        if save_pdf:
            plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
        if save_eps:
            plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
        if save_svg:
            plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
        if save_jpg:
                plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
        if plt_show:
            plt.show()
        plt.close(fig)


def plot_carla_MARL_exp_timesteps_A2A_connections(results_MARL, results_ICP, env, params, fontsize = 18, labelsize = 18, save_eps = 1, save_svg = 1, save_pdf = 1, save_jpg = 1, plt_show = 1):
    

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GPS'] + ['ICP'] + ['ICP-MAPPO single'] + ['ICP-MAPPO']
    agents_markers = [2, 10, 19]
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    temp_marker_style_indexes_per_method = ['o', 's', '^', 'D']  # Example marker styles: circle, square, triangle up, diamond
    marker_style_indexes_per_method = {marker: temp_marker_style_indexes_per_method[i] for i, marker in enumerate(agents_markers)}


    ICP_num_agents_connections = {num_agents: np.sum(np.sum(v['connectivity_matrix_A2A'], 1), 1) for num_agents, v in results_ICP.items()}
    MARL_single_A_absolute_error_pos = {num_agents: np.moveaxis(np.array([v[step]['MARL_no_coop_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents)
    MARL_A_absolute_error_pos = {num_agents: np.moveaxis(np.array([v[step]['MARL_absolute_error_pos'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents)
    
    try: # MC > 1
        MARL_single_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 2)).squeeze() for num_agents, v in MARL_single_A_absolute_error_pos.items()} # (max_num_agents, timestep, MC)
        MARL_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 2)).squeeze() for num_agents, v in MARL_A_absolute_error_pos.items()} # (max_num_agents, timestep, MC)
    except: # MC = 1
        MARL_single_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_agents, v in MARL_single_A_absolute_error_pos.items()} # (max_num_agents, timestep, 1)
        MARL_A_RMSE_pos = {num_agents: np.sqrt(np.mean((v**2), 0)).squeeze().reshape(-1, 1) for num_agents, v in MARL_A_absolute_error_pos.items()} # (max_num_agents, timestep, 1)

    try: # MC > 1
        MARL_total_actions_from_policy = {num_agents: np.moveaxis(np.array([v[step]['total_actions'] for step in range(len(v))]).squeeze(), 0, 1) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, MC, num_agents, 1)
        MARL_num_agents_from_policy = {num_agents: np.sum(v, 2) for num_agents, v in MARL_total_actions_from_policy.items()} # (max_num_agents, timestep, MC)
    except:
        MARL_total_actions_from_policy = {num_agents: np.array([v[step]['total_actions'] for step in range(len(v))]).reshape(-1, 1, params.num_agents) for num_agents, v in results_MARL.items()} # (max_num_agents, timestep, 1, num_agents)
        MARL_num_agents_from_policy = {num_agents: np.sum(v, 2) for num_agents, v in MARL_total_actions_from_policy.items()} # (max_num_agents, timestep, 1)

    timestep = list(MARL_total_actions_from_policy.values())[0].shape[0]
    MC = list(MARL_total_actions_from_policy.values())[0].shape[1]
    num_max_agents = list(MARL_A_RMSE_pos.keys())



    # NUM CONNECTIONS vs TIMESTEP
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    # MARL 
    method = 'ICP-MAPPO'
    for num_agents in agents_markers:
        data = np.squeeze(MARL_num_agents_from_policy[list(MARL_num_agents_from_policy.keys())[num_agents]]) # (num_max_agents, timestep)
        
        mean = np.mean(data, 0)
        std = np.std(data, 0)[:len(mean)] 
        
        ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]], marker=marker_style_indexes_per_method[num_agents], markevery=40)
        ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[color_indexes_per_method[method]], alpha=0.2)

    # ICP 
    method = 'ICP'
    for num_agents in agents_markers:
        data = np.squeeze(ICP_num_agents_connections[list(ICP_num_agents_connections.keys())[num_agents]]) # (num_max_agents, timestep)

        mean = np.mean(data, 0)
        std = np.std(data, 0)[:len(mean)] 

        ax.plot(mean, linestyle=line_style_indexes_per_method[method], color=colors[color_indexes_per_method[method]], marker=marker_style_indexes_per_method[num_agents], markevery=40)


    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.30)
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # ax.set_xlim([0, 20])
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
    plt.ylabel('Number of connections', fontsize=fontsize)     
    plt.xlabel('Timestep', fontsize=fontsize)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    # Legends
    legend_handles = []
    for method in legend_labels:
        for num_agents in agents_markers:
            if method in ['ICP-MAPPO', 'ICP']:
                line = mlines.Line2D([], [], color=colors[color_indexes_per_method[method]], linestyle=line_style_indexes_per_method[method], label=method + ' {} agents'.format(num_agents+1 if num_agents == 19 else num_agents), marker=marker_style_indexes_per_method[num_agents])
                legend_handles.append(line)

    ax.legend(handles=legend_handles, loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'Connections_per_timestep'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

    # BARS (NUM CONNECTIONS) vs NUM MAX AGENTS (FOR ICP-MAPPO AND ICP)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    bar_width = 0.7  # Width of the bars
    bar_positions = np.array(agents_markers)

    # MARL (ICP-MAPPO)
    res_ICP_MAPPO = {}
    method = 'ICP-MAPPO'
    for i, num_agents in enumerate(agents_markers):
        data = np.squeeze(MARL_num_agents_from_policy[list(MARL_num_agents_from_policy.keys())[num_agents]]) # (num_max_agents, timestep)
        
        mean = np.mean(data, 0)
        std = np.std(data, 0)[:len(mean)] 

        ax.bar(bar_positions[i] - bar_width/2, mean, yerr=std, width=bar_width, color=colors[color_indexes_per_method[method]], alpha=0.5, label=method if i == 0 else "")
        res_ICP_MAPPO[num_agents] = [mean, std]
    
    # ICP
    res_ICP = {}
    method = 'ICP'
    for i, num_agents in enumerate(agents_markers):
        data = np.squeeze(ICP_num_agents_connections[list(ICP_num_agents_connections.keys())[num_agents]]) # (num_max_agents, timestep)
        
        mean = np.mean(data, 0)
        std = np.std(data, 0)[:len(mean)] 

        ax.bar(bar_positions[i] + bar_width/2, mean, yerr=std, width=bar_width, color=colors[color_indexes_per_method[method]], alpha=0.5, label=method if i == 0 else "")
        res_ICP[num_agents] = [mean, std]

    plt.xticks([2, 6, 10, 15, 20], [2, 6, 10, 15, 20], rotation=0, ha='center')  # Set x-axis ticks to show 2, 10, 20
    plt.subplots_adjust(bottom=0.30)
    plt.ylabel('Number of connections', fontsize=fontsize)
    plt.xlabel('Number of agents', fontsize=fontsize)
    plt.tick_params(labelsize=labelsize)

    # Legends
    ax.legend(loc='best', shadow=False, fontsize=fontsize)

    log_path = params.Figures_dir
    file_name = 'Connections_per_num_agents'
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
        plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)




        

##############################################################################################################################
############# CARLA SCENARIO ######################################################################
##############################################################################################################################


def draw_vehicle_features (params, env, dataset, colors = None, type_= None, which_edges = None, gt = None, image_path = None, plot_fake_vehicles = None, vehicles_pos_xy=None, DIR_EXPERIMENT=None, save_eps = None, title_savename = None):
    
    plt.rcParams.update({'font.size': 20})
    dataset = {dataset_output_name:dataset[index_name] for index_name,(dataset_output_name) in enumerate(env.dataset_output_names)}

    instant = 100

    G_vehicle_features = nx.Graph()


    # Find all vehicles names
    vehicles = [f'vehicle {v}' for v in range(params.num_agents)] 
    name_features = [f'pole {v}' for v in range(params.num_features)]

            
    # vehicles_pos_xy and add nodes
    pos = {}
    for i, v in enumerate(vehicles):
        # print(v, instant)
        pos[v] = tuple([-dataset['t_A'][instant][i][1],dataset['t_A'][instant][i][0]]) # ACCORDING TO REF SYS
        G_vehicle_features.add_node(v)
        # print(v, pos[v])
    for j, f in enumerate(name_features):
        # mean of centroids 
        pos[f] = tuple([-dataset['t_F'][instant][j][1], dataset['t_F'][instant][j][0] ])  # ACCORDING TO REF SYS
        G_vehicle_features.add_node(f)
        
    # Add edges
    edges_to_add = []
    for i, v in enumerate(vehicles):
        for j, f in enumerate(name_features):
            if dataset['connectivity_matrix_A2F'][instant][i][j] == 1:
                edges_to_add.append(tuple([v, f]))
    G_vehicle_features.add_edges_from(edges_to_add)
        
    # PLOTTING
    fig = plt.figure(figsize=(15, 15))
           
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(vehicles))]
    color = np.array([colors[int(i)] for i, vehicle in enumerate(vehicles)])
        
    # features predicted 
    nx.draw_networkx(G_vehicle_features, pos, nodelist=name_features, edgelist=[], node_shape="o", node_size=100, alpha=0.75, node_color='red', with_labels=False)   
    # vehicles
    # node_color=color for each vehicle a different color
    nx.draw_networkx(G_vehicle_features, pos, nodelist=vehicles, edgelist=[], node_shape="s", node_size=100, alpha=1, node_color='blue', with_labels=False)
        
    # plot edges
    edges_to_draw = list(G_vehicle_features.edges())
          
    nx.draw_networkx(
        G_vehicle_features,
        pos,
        edgelist = edges_to_draw,
        nodelist=[],
        width=1,# 3
        alpha=0.5,
        edge_color="black", #"tab:gray",
        with_labels=False
    )

    plt.axis("equal")

    # legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
    #                     Line2D([0], [0], color="blue", lw=4, label='Targets', marker='^', markersize = 10, linestyle="None")]
    # legend1 = plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(+1, +1.31))#(+1, +1.16))
    # elements = []
    # for i, vehicle in enumerate(vehicles):
    #     elements.append(Line2D([0], [0], color=colors[int(i)], lw=4,  marker='s', markersize = 10, linestyle="None", label=f'Vehicle{vehicle}'))
    # legend2 = plt.legend(handles=elements, loc='upper left', ncol = 3, bbox_to_anchor=(-0.005, +1.31))#(-0.1, +1.16))
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    
    # Import Map
    img = plt.imread(image_path)
    # img = Image.open(image_path)
    y_shift = 0# +95 # 0 # +95
    x_shift = 0# -205 # 0#-205

    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    # dx, dy = 100,100
    # # Custom (rgb) grid color
    # grid_color = [0,0,0,0]
    # # Modify the image to include the grid
    # img[:,::dy,:] = grid_color
    # img[::dx,:,:] = grid_color

    limit_xl = -313
    limit_xr = -100
    limit_yl = -14
    limit_yr = 200
    
    x_values = np.arange (limit_xl, limit_xr, (limit_xr-limit_xl)/9)
    y_values = np.arange (limit_yl, limit_yr, (limit_yr-limit_yl)/9)
    # for x_ in x_values:
    #     for y_ in y_values:
    #         if (x_ == limit_xl or x_ == limit_xr) and (x_!=limit_xl or y_!=limit_yr) and (x_!=limit_xl or y_!=limit_yl) and (x_!=limit_xr or y_!=limit_yr) and (x_!=limit_xr or y_!=limit_yl):
    #             plt.text(x_,y_,f'{y_:.1f}',color='k',ha='center',va='center')  
    #         if (y_ == limit_yl or y_ == limit_yr) and (x_!=limit_xl or y_!=limit_yr) and (x_!=limit_xl or y_!=limit_yl) and (x_!=limit_xr or y_!=limit_yr) and (x_!=limit_xr or y_!=limit_yl):
    #             plt.text(x_,y_,f'{x_:.1f}',color='k',ha='center',va='center') 
              
    # plt.text(limit_xl,limit_yr,'(-314,200)',color='k',ha='center',va='center')
    # plt.text(limit_xl,limit_yl,'(-314,-14)',color='k',ha='center',va='center')
    # plt.text(limit_xr,limit_yr,'(-100,200)',color='k',ha='center',va='center')
    # plt.text(limit_xr,limit_yl,'(-100,-14)',color='k',ha='center',va='center')
    plt.imshow(img, extent=[limit_xr+x_shift, limit_xl+x_shift, limit_yl+y_shift, limit_yr+y_shift]) # from the edges of thr road 

    
    log_path = params.Figures_dir
    file_name = 'Scenario'

    plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)

    # plt.show()
    # plt.close(fig)

    plt.rcParams.update({'font.size': 22})  
   
    return colors