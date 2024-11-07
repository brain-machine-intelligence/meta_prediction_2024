import analysis
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import getopt
import sys
import csv
import os
import analysis
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import torch
import os
import simulation as sim
import math
from statistics import stdev
from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
#from training_layer2 import Net
from torch.autograd import Variable
import analysis
from scipy import stats
from scipy.interpolate import pchip_interpolate
from analysis import save_plt_figure
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import scipy.io
from math import ceil

tsne_go = False

#action_list = ['Nill','R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']  # , 'Uncertainty']
NUM_ACT =len(action_list)
pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
pol_list_tick = ['minR(LR)','maxR(HR)','minS(LS)','maxS(HS)','LRLS','HRHS','LRHS','HRLS']

file_suffix = '_20215_20_trials_20240725_delta_control_restore_drop_highest'
#file_suffix = '_20201_20_trials_20240725_delta_control_restore_drop_highest'
#file_suffix = '_2023_20_trials_20231214_delta_control_restore_drop_highest'
file_suffix = '_2020_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_2021_20_trials_20240717_delta_control_restore_drop_highest'
#file_suffix = '_2020_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_2014_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_20211_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20212_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20213_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20214_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20215_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20201_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20202_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20203_20_trials_20240110ab_delta_control_restore_drop_highest'
print(file_suffix)

if '20240725_delta_control_restore_drop' in file_suffix:
    folderpath = '20240725' #SPE task with transition probability action change ablation
    if '2021' in file_suffix:
        if '20211' in file_suffix:
            action_list = ['0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
        elif '20212' in file_suffix:
            action_list = ['Nill', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
        elif '20213' in file_suffix:
            action_list = ['Nill', '0.5<->0.9', 'R-recover-visited', 'R-recover-unvisit']
        elif '20214' in file_suffix:
            action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-unvisit']
        elif '20215' in file_suffix:
            action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited']
        else:
            print('task type error')
        action_list_column = ['action_0', 'action_1', 'action_2', 'action_3']
    elif '2020' in file_suffix:
        if '20201' in file_suffix:
            action_list = ['R-recover-visited', 'R-recover-unvisit']
        elif '20202' in file_suffix:
            action_list = ['Nill', 'R-recover-unvisit']
        elif '20203' in file_suffix:
            action_list = ['Nill', 'R-recover-visited']
        else:
            print('task type error')
        action_list_column = ['action_0', 'action_1']
    else:
        print('error in task type')
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix == '_2021_20_trials_20240717_delta_control_restore_drop_highest':
    folderpath = '20240717' #SPE task with transition probability action change ablation
    action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
    action_list_column = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
    pol_list = ['min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix == '_2023_20_trials_20231214_delta_control_restore_drop_highest':
    folderpath = '20231214' #SPE task with transition probability action change ablation
    action_list = ['Nill', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
    action_list_column = ['action_0', 'action_1', 'action_2', 'action_3']
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix == '_2021_20_trials_20231214_delta_control_restore_drop_highest':
    folderpath = '20231214' #previous SPE tasks
    action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']  # , 'Uncertainty']
    action_list_column = ['action_0', 'action_1', 'action_2', 'action_3',
                          'action_4']  # , 'Uncertainty']file_suffix = '_2021_20_trials_20221010_delta_control_highest' #2021 task + 10000 eps_
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB','min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix == '_2020_20_trials_20231214_delta_control_restore_drop_highest':
    folderpath = '20231214'  # previous RPE tasks
    action_list = ['Nill', 'R-recover-visited', 'R-recover-unvisit']  # , 'Uncertainty']
    action_list_column = ['action_0', 'action_1', 'action_2']
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['minR(LR)', 'maxR(HR)', 'minS(LS)', 'maxS(HS)', 'LRLS', 'HRHS', 'LRHS', 'HRLS']
elif file_suffix == '_2014_20_trials_20231214_delta_control_restore_drop_highest':
    folderpath = '20231214'  # previous ori tasks
    action_list = ['Nill', '0.5<->0.9', 'S<->F']  # , 'Uncertainty']
    action_list_column = ['action_0', 'action_1', 'action_2']
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix in ['_20201_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20202_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20203_20_trials_20240110ab_delta_control_restore_drop_highest']:
    folderpath = '20240110ab'  # previous ori tasks
    if file_suffix == '_20201_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['R-recover-visited', 'R-recover-unvisit']
    elif file_suffix == '_20202_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', 'R-recover-unvisit']
    elif file_suffix == '_20203_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', 'R-recover-visited']
    action_list_column = ['action_0', 'action_1']
    pol_list = ['min-rpe', 'max-rpe']
    pol_list_tick = ['r', 'R']
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix in ['_20211_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20212_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20213_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20214_20_trials_20240110ab_delta_control_restore_drop_highest',
                     '_20215_20_trials_20240110ab_delta_control_restore_drop_highest']:
    folderpath = '20240110ab'  # previous ori tasks
    if file_suffix == '_20211_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
    elif file_suffix == '_20212_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']
    elif file_suffix == '_20213_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', '0.5<->0.9', 'R-recover-visited', 'R-recover-unvisit']
    elif file_suffix == '_20214_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-unvisit']
    elif file_suffix == '_20215_20_trials_20240110ab_delta_control_restore_drop_highest':
        action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited']
    action_list_column = ['action_0', 'action_1', 'action_2', 'action_3']
    pol_list = ['min-spe', 'max-spe']
    pol_list_tick = ['s', 'S']
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
elif file_suffix == '-_2020_20_trials_20240110ab_delta_control_restore_drop':
    folderpath = '20240105'  # previous ori tasks
    folderpath = '20240110ab'  # previous ori tasks
    action_list = ['Nill', '0.5<->0.9', 'S<->F']  # , 'Uncertainty']
    action_list_column = ['action_0', 'action_1', 'action_2']
    pol_list = ['min-rpe', 'max-rpe']
    pol_list_tick = ['r', 'R']
elif file_suffix == '-_2021_20_trials_20240105_delta_control_restore_drop':
    folderpath = '20240105'  # previous ori tasks
    action_list = ['Nill', '0.5<->0.9', 'S<->F', 'R-recover-visited', 'R-recover-unvisit']  # , 'Uncertainty']
    action_list_column = ['action_0', 'action_1', 'action_2', 'action_3',
                          'action_4']
    pol_list = ['min-spe', 'max-spe']
    pol_list_tick = ['s', 'S']
NUM_ACT = len(action_list)
NUM_POL = len(pol_list)
print(file_suffix)

COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
DATA_COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward',
                        'score'] + action_list_column + ['applied_reward']
DETAIL_COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score','action'] \
                 + ['rpe1', 'rpe2','spe1', 'spe2','0', '10', '20', '40', 'visit', 'applied_reward'] \
                 + ['ACT1','ACT2','STT1','STT2','trans_prob','bwd_idf']
#COLUMNS = ['rpe','spe','ctrl_reward','score','0','10','20','40','visit', 'applied_reward','rpe1','rpe2','spe1','spe2']
#COLUMNS = ['ctrl_reward']
pickle_filename='history_results/' + folderpath + '/Analysis-Object-' + pol_list[0] + '-00' + file_suffix + '.pkl'
with open(pickle_filename,'rb') as f:
    data = pickle.load(f)
    NUM_EPISODES, NUM_FULL_FEATS_DATA = data.data[pol_list[0]][0].shape
    NUM_FULL_TRIALS, NUM_FULL_FEATS_DETAIL = data.detail[pol_list[0]][0].shape
    TRIALS_PER_EPISODE = ceil(NUM_FULL_TRIALS / NUM_EPISODES)
max_indx = 82
#TRIALS_PER_EPISODE = 20
#NUM_EPISODES = 1000
full_data = np.zeros((len(DATA_COLUMNS),NUM_POL,max_indx,NUM_EPISODES))
full_detail = np.zeros((len(DETAIL_COLUMNS),NUM_POL,max_indx,TRIALS_PER_EPISODE*NUM_EPISODES))
full_opt = np.zeros((len(DETAIL_COLUMNS),NUM_POL,max_indx))
full_trials_opt = np.zeros((len(DETAIL_COLUMNS),NUM_POL,max_indx,TRIALS_PER_EPISODE))
real_opt = np.zeros((len(DETAIL_COLUMNS),NUM_POL,max_indx))
full_plot = np.zeros((len(DETAIL_COLUMNS),NUM_POL,2))
full_opt_plot = np.zeros((len(DETAIL_COLUMNS),NUM_POL,2))
real_opt_plot = np.zeros((len(DETAIL_COLUMNS),NUM_POL,2))
full_SBJ_plot = np.zeros((len(DETAIL_COLUMNS),max_indx,2))
opt_index= np.zeros((NUM_POL,max_indx))
opt_pol = np.zeros((NUM_POL,max_indx,TRIALS_PER_EPISODE ,1))
RPE_plot_detail = np.zeros((NUM_POL,2))
SPE_plot_detail = np.zeros((NUM_POL,2))
PMB_plot_detail = np.zeros((NUM_POL,2))
RWD_plot_detail = np.zeros((NUM_POL,2))
save_pol = np.zeros((NUM_POL,TRIALS_PER_EPISODE ,2))
time_delay = [1.7, 4.06, 6.99]
single_game_time = 8.4
gen_regressor = False
TR = 2.8
full_detail_expanded = np.zeros((ceil(max_indx*NUM_POL*0.5*NUM_EPISODES),ceil(TRIALS_PER_EPISODE *single_game_time/TR)))
for pol in range(len(pol_list)):
    print(pol_list[pol])
    for sbj in range(max_indx):
        #print(np.load('history_results/' + folderpath + '/final_opt_pol_' + pol_list[pol] + '_{0:02d}'.format(
        #    sbj) + '_2021_20_trials_delta_control' + '.npy'))
        with open('history_results/' + folderpath + '/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl','rb') as f:
            #print('history_results/' + folderpath + '/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl')
            data = pickle.load(f)
            #data.current_data: eps, data.current_detial : eps*trials
        for feat_indx in range(len(DATA_COLUMNS)):
            # if feat_indx == 0: full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
            #print(data.data[pol_list[pol]][0])
            full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][DATA_COLUMNS[feat_indx]]
            opt_index[pol][sbj]=data.data[pol_list[pol]][0]['ctrl_reward'].loc[ceil(0.1  * len(data.data[pol_list[pol]][0])):].idxmax()
            #print(data.data[pol_list[pol]][0]['ctrl_reward'][opt_index[pol][sbj]])
        for feat_indx in range(len(DETAIL_COLUMNS)):
            #print(data.detail[pol_list[pol]][0][DETAIL_COLUMNS[feat_indx]])
            full_detail[feat_indx][pol][sbj] = data.detail[pol_list[pol]][0][DETAIL_COLUMNS[feat_indx]]
            for t_indx in range(TRIALS_PER_EPISODE ):
                opt_pol[pol][sbj][t_indx] = data.detail[pol_list[pol]][0]['action'].loc[
                                      opt_index[pol][sbj]*TRIALS_PER_EPISODE -TRIALS_PER_EPISODE +t_indx]
                # opt_pol[pol][sbj][t_indx][1] = data.detail[pol_list[pol]][0]['action'].loc[opt_index[pol][sbj] * TRIALS_PER_EPISODE  - TRIALS_PER_EPISODE  + t_indx][1]
            full_opt[feat_indx][pol][sbj] = sum(full_detail[feat_indx][pol][sbj]
                                                [int(TRIALS_PER_EPISODE *opt_index[pol][sbj]-TRIALS_PER_EPISODE ):
                                                 int(TRIALS_PER_EPISODE *opt_index[pol][sbj])]) / TRIALS_PER_EPISODE
            full_trials_opt[feat_indx][pol][sbj] = full_detail[feat_indx][pol][sbj][int(TRIALS_PER_EPISODE *opt_index[pol][sbj]-TRIALS_PER_EPISODE ):
                                                 int(TRIALS_PER_EPISODE *opt_index[pol][sbj])]
            full_opt_plot[feat_indx][pol][0] += np.mean(full_opt[feat_indx][pol][sbj])/max_indx
            real_opt[feat_indx][pol][sbj] = sum(full_detail[feat_indx][pol][sbj]
                                                [int(TRIALS_PER_EPISODE * opt_index[pol][sbj] - ceil(TRIALS_PER_EPISODE /2)):
                                                 int(TRIALS_PER_EPISODE * opt_index[pol][sbj])]) / TRIALS_PER_EPISODE
            real_opt_plot[feat_indx][pol][0] += np.mean(real_opt[feat_indx][pol][sbj])/max_indx
            full_plot[feat_indx][pol][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.5*TRIALS_PER_EPISODE*NUM_EPISODES):])/max_indx
            full_SBJ_plot[feat_indx][sbj][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.5*TRIALS_PER_EPISODE*NUM_EPISODES):])/4
        for ep_indx in range(ceil(0.2*NUM_EPISODES+1),NUM_EPISODES):
            for trial_indx in range(TRIALS_PER_EPISODE ):
                rpe1_tindx = ceil((trial_indx * single_game_time + time_delay[1]) / TR)
                rpe2_tindx = ceil((trial_indx * single_game_time + time_delay[2]) / TR)
                if gen_regressor == True:
                    full_detail_expanded[ceil(pol * max_indx * 0.8 *TRIALS_PER_EPISODE + sbj * 0.8 *TRIALS_PER_EPISODE + ep_indx
                                         - 0.2 *TRIALS_PER_EPISODE)][rpe1_tindx] = \
                                        full_detail[1][pol][sbj][ep_indx * TRIALS_PER_EPISODE  + trial_indx]
                    full_detail_expanded[ceil(pol * max_indx * 0.8 *TRIALS_PER_EPISODE + sbj * 0.8 *TRIALS_PER_EPISODE + ep_indx
                                         - 0.8 *TRIALS_PER_EPISODE)][rpe2_tindx] = \
                                        full_detail[2][pol][sbj][ep_indx * TRIALS_PER_EPISODE  + trial_indx]
#    RPE_plot[pol][0] = sum(opt_RPE[pol])/len(opt_RPE[pol])
 #   SPE_plot[pol][0] = sum(opt_SPE[pol]) / len(opt_SPE[pol])
    for feat_indx in range(len(COLUMNS)):
        full_plot[feat_indx][pol][1] = stdev(full_opt[feat_indx][pol])/np.sqrt(len(full_opt[feat_indx][pol]))
        full_opt_plot[feat_indx][pol][1] = stdev(full_opt[feat_indx][pol]) / np.sqrt(len(full_opt[feat_indx][pol]))
        real_opt_plot[feat_indx][pol][1] = stdev(real_opt[feat_indx][pol]) / np.sqrt(len(real_opt[feat_indx][pol]))
    save_pol[pol] = opt_pol[pol][81]
    scipy.io.savemat('history_results/' + folderpath + '/Policy result in the ' + pol_list[pol] + 'data'+ file_suffix + '.mat',
                     {'data': opt_pol[pol]})


for sbj in range(max_indx):
    for feat_indx in range(len(COLUMNS)):
        tmp = np.transpose(full_opt[feat_indx])
        full_SBJ_plot[feat_indx][sbj][1] = stdev(tmp[sbj])/np.sqrt(len(pol_list))

np.save('history_results/' + folderpath + '/optimal_policy'+file_suffix+'.npy',opt_pol)
np.save('history_results/' + folderpath + '/feat'+file_suffix+'.npy',full_detail)
print(full_opt.shape)
np.save('history_results/' + folderpath + '/feat_full_opt'+file_suffix+'.npy',full_opt)
np.save('history_results/' + folderpath + '/feat_full_opt_full_trials'+file_suffix+'.npy',full_trials_opt)
np.save('history_results/' + folderpath + '/optimal_policy_index'+file_suffix+'.npy',opt_index)
scipy.io.savemat('history_results/' + folderpath + '/Policy result in the full data'+file_suffix+'.mat', {'data': opt_pol})
#if gen_regressor == True:
#   scipy.io.savemat('history_results/RPE_regressor'+file_suffix+'.mat',{'RPE': full_detail_expanded})

for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(NUM_POL)
    temp_std = np.zeros(NUM_POL)
    print(COLUMNS[feat_indx])
    for ii in range(NUM_POL):
        temp_mean[ii]=full_plot[feat_indx][ii][0]
        temp_std[ii] = full_plot[feat_indx][ii][1]
        print(pol_list[ii] + " " + str(temp_mean[ii]) + " " + str(temp_std[ii]))
        if ii%2 == 1:
            #full_detail = np.zeros((len(DETAIL_COLUMNS), NUM_POL, max_indx, TRIALS_PER_EPISODE * NUM_EPISODES))
            #real_opt[feat_indx][pol][sbj]
            #t_stat, p_val = stats.ttest_ind(full_detail[feat_indx][ii].flatten(),full_detail[feat_indx][ii-1].flatten())
            t_stat, p_val = stats.ttest_ind(real_opt[feat_indx][ii],real_opt[feat_indx][ii-1])
            print(pol_list[ii-1] + " " + pol_list[ii] + " p " + str(p_val))
    plt.bar(range(NUM_POL), temp_mean)
    plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
    plt.xticks(range(NUM_POL), pol_list_tick)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/' + folderpath + '/'+COLUMNS[feat_indx]+'_plot' + file_suffix + '.png')
    plt.clf()

for feat_indx in range(len(COLUMNS)):
    for mode_idf in range(len(pol_list)):
        #plt.plot(np.transpose(full_detail[feat_indx][mode_idf]))
        #plt.savefig('history_results/'+COLUMNS[feat_indx]+'_'+pol_list[mode_idf]+'_full_plot' + file_suffix + '.png')
        #plt.clf()
        plt.plot(np.transpose(full_data[feat_indx][mode_idf]))
        plt.savefig(
            'history_results/' + folderpath + '/' + COLUMNS[feat_indx] + '_' + pol_list[mode_idf] + '_abstract_plot' + file_suffix + '.png')
        plt.clf()

for pol in range(NUM_POL):
    print(pol_list[pol])
    pol_acts = np.zeros((NUM_ACT, TRIALS_PER_EPISODE ))
    for ii in range(82):
        for t_indx in range(TRIALS_PER_EPISODE):
            pol_acts[int(opt_pol[pol][ii][t_indx][0])][t_indx] += 1

    pol_acts /= 82

    for ii in range(len(pol_acts)):
        plt.plot(pol_acts[ii], label=ii)
    plt.legend(action_list, loc=5)
    plt.ylabel('Action Frequency')
    plt.xlabel('Episode')
    plt.title('Action frequency in the '+pol_list[pol]+' optimal sequences')
    plt.ylim((0, 1))
    plt.savefig('history_results/' + folderpath + '/Action_frequency_'+pol_list[pol] + file_suffix +'_opt.png')
    plt.clf()

for pol in range(NUM_POL):
    print(pol_list[pol])
    pol_acts2 = np.zeros((NUM_ACT, max_indx))
    for ii in range(82):
        for t_indx in range(TRIALS_PER_EPISODE ):
            pol_acts2[int(opt_pol[pol][ii][t_indx][0])][ii] += 1
    print('pol_dist')

    for ii in range(NUM_ACT):
        temp_script = ""
        temp_script = temp_script + "action" + str(ii) + ": " + str(np.mean(pol_acts2[ii])) + " " + str(stdev(pol_acts2[ii]) / np.sqrt(82)) + " "
        print(temp_script)



for feat_indx in range(len(COLUMNS)):
    temp_mean = np.zeros(NUM_POL)
    temp_std = np.zeros(NUM_POL)
    for ii in range(NUM_POL):
        temp_mean[ii]=full_plot[feat_indx][ii][0]
        temp_std[ii] = full_plot[feat_indx][ii][1]
    plt.bar(range(NUM_POL), temp_mean)
    plt.errorbar(range(NUM_POL), temp_mean, yerr=temp_std)
    plt.xticks(range(NUM_POL), pol_list_tick)
    plt.title(COLUMNS[feat_indx])
    plt.savefig('history_results/' + folderpath + '/' +COLUMNS[feat_indx]+'_plot' + file_suffix + '.png')
    plt.clf()

temp_mean = np.zeros((NUM_POL,NUM_EPISODES))
temp_std = np.zeros((NUM_POL,NUM_EPISODES))
print(DATA_COLUMNS[6])
for ii in range(NUM_POL):
    for jj in range(NUM_EPISODES):
        #print(np.shape(full_data[6,ii,:,jj]))
        temp_mean[ii][jj] = np.mean(full_data[6,ii,:,jj])
        #temp_std[ii][jj] = stdev(full_data[6,ii,:,jj])/np.sqrt(len(full_data[6,ii,:,jj]))
        temp_std[ii][jj] = stdev(full_data[6, ii, :, jj])

    TICK_NAME_NUM = np.linspace(1, NUM_EPISODES + 1, NUM_EPISODES + 1)[:-1]
    TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
    smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
    smooth_y = pchip_interpolate(TICK_NAME_NUM, temp_mean[ii], smooth_x)
    ax = plt.gca()
    ax.plot(smooth_x, smooth_y, color='red')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Controller reward for ' + pol_list[ii])
    smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, temp_std[ii], smooth_x))
    ax.fill_between(smooth_x, smooth_y - 1.96 * smooth_sem, smooth_y + 1.96 * smooth_sem, alpha=0.2,
                    color='red')
    plt.legend(bbox_to_anchor=(1.02, 1))
    plt.savefig(
        'history_results/' + folderpath + '/' + 'Averaged training controller reward learning curve result in the ' +
        pol_list[ii]+ file_suffix + '.png')

    plt.clf()


print(DATA_COLUMNS[6])
for pol in range(NUM_POL):
    for sbj in range(max_indx):
        with open('history_results/Analysis-Object-' + pol_list[pol] + '-{0:02d}'.format(sbj) + file_suffix[:-8] + '.pkl',
                  'rb') as f:
            data = pickle.load(f)
        #print(pol_list[pol] + ' ' + str(sbj))
        TICK_NAME_NUM = np.linspace(1, len(data.data[pol_list[pol]][0]['ctrl_reward']) + 1, len(data.data[pol_list[pol]][0]['ctrl_reward']) + 1)[:-1]
        TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
        smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
        smooth_y = pchip_interpolate(TICK_NAME_NUM, data.data[pol_list[pol]][0]['ctrl_reward'], smooth_x)
        ax = plt.gca()
        ax.plot(smooth_x, smooth_y, color='red')
        plt.savefig(
            'history_results/' + folderpath + '-sbj/' + 'Averaged training controller reward learning curve result in the ' +
            pol_list[pol]+ file_suffix + '_sbj' + str(sbj) + '.png')

        plt.clf()


print(file_suffix)

'''
plt.clf()
ori_mean = np.zeros(19)
ori_sem = np.zeros(19)
RT_std = scipy.io.loadmat('RT_std.mat')
RT_mean = scipy.io.loadmat('RT_mean.mat')
ori_mean = RT_mean['RT_mean'][0]
ori_sem = RT_std['RT_std'][0]
TICK_NAME_NUM = np.linspace(1, 19 + 1, 19+1)[:-1]
TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
smooth_x = np.linspace(1, TICK_NAME_NUM[-1], 200)
smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
ax = plt.gca()
ax.plot(smooth_x, smooth_y, label='RPE min', color='grey')
ax.set_xlabel('Game step in one policy')
ax.set_ylabel('Reaction time (s)')
smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
ax.fill_between(smooth_x, smooth_y - 1.96*smooth_sem, smooth_y + 1.96*smooth_sem, alpha=0.2,
                color='grey')

ori_mean = RT_mean['RT_mean'][1]
ori_sem = RT_std['RT_std'][1]
TICK_NAME_NUM = np.linspace(1, 19 + 1, 19 + 1)[:-1]
TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
smooth_x = np.linspace(1, TICK_NAME_NUM[-1], 200)
smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
ax = plt.gca()
ax.plot(smooth_x, smooth_y, label='RPE max', color='black')
smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
ax.fill_between(smooth_x, smooth_y - 1.96*smooth_sem, smooth_y + 1.96*smooth_sem, alpha=0.2,
                color='black')
plt.savefig('RPE_RT_without_legend.png')                
plt.legend(bbox_to_anchor=(1.02, 1))
plt.savefig('RPE_RT.png')

plt.clf()
ori_mean = RT_mean['RT_mean'][2]
ori_sem = RT_std['RT_std'][2]
TICK_NAME_NUM = np.linspace(1, 19 + 1, 19+1)[:-1]
TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
smooth_x = np.linspace(1, TICK_NAME_NUM[-1], 200)
smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
ax = plt.gca()
ax.plot(smooth_x, smooth_y, label='SPE min', color='grey')
smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
ax.fill_between(smooth_x, smooth_y - 1.96*smooth_sem, smooth_y + 1.96*smooth_sem, alpha=0.2,
                color='grey')

ori_mean = RT_mean['RT_mean'][3]
ori_sem = RT_std['RT_std'][3]
TICK_NAME_NUM = np.linspace(1, 19 + 1, 19 + 1)[:-1]
TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
smooth_x = np.linspace(1, TICK_NAME_NUM[-1], 200)
smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
ax = plt.gca()
ax.plot(smooth_x, smooth_y, label='SPE max', color='black')
ax.set_xlabel('Game step in one policy')
ax.set_ylabel('Reaction time (s)')
smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
ax.fill_between(smooth_x, smooth_y - 1.96*smooth_sem, smooth_y + 1.96*smooth_sem, alpha=0.2,
                color='black')
plt.savefig('SPE_RT_without_legend.png')
plt.legend(bbox_to_anchor=(0.4, 1))
plt.savefig('SPE_RT.png')
'''
