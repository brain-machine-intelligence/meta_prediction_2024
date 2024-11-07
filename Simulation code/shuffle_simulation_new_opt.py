import torch
import getopt
import sys
import csv
import os
import simulation as sim
import math
from random import randint
from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
from torch.autograd import Variable
import pandas as pd
import analysis

import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
from analysis import save_plt_figure

from mdp import MDP
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import analysis
from math import ceil
import scipy.io as sio
#from main import MODE_LIST
# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 200
TRIALS_PER_EPISODE    = 80
SPE_LOW_THRESHOLD     = 0.3#0.3
SPE_HIGH_THRESHOLD    = 0.45#0.5
RPE_LOW_THRESHOLD     = 4
RPE_HIGH_THRESHOLD    = 9 #10
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD  = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD  = 0.3
CONTROL_REWARD        = 1
CONTROL_REWARD_BIAS   = 0
INIT_CTRL_INPUT       = [10, 0.5]
DEFAULT_CONTROL_MODE  = 'max-spe'
CONTROL_MODE          = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED   = True
RPE_DISCOUNT_FACTOR   = 0.003
ACTION_PERIOD         = 3
STATIC_CONTROL_AGENT  = False
ENABLE_PLOT           = True
DISABLE_C_EXTENSION   = False
LEGACY_MODE           = False
MORE_CONTROL_INPUT    = True
SAVE_CTRL_RL          = False
PMB_CONTROL = False
TASK_TYPE = 2020
MF_ONLY = False
MB_ONLY = False
Reproduce_BHV = False
saved_policy_path = ''
Session_block = False
mode202010 = False
DECAY_RATE = 0.75
RESTORE_DROP_RATE = 0.0
turn_off_tqdm = False
CONTROL_resting = 99 #Intial duration for CONTROL agent resting
max_sbj = 82

RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']
folderpath = '20210827' # old
folderpath = '20230627' # Restore drop out as task
folderpath = '20230926' # Restore drop out as task
folderpath = '20230927' # Restore drop out as task
folderpath = '20231020' # 2014 task setting
folderpath = '20231214' # 2014 task setting
folderpath = '202405222'

error_reward_map = {
    # x should be a 4-tuple: rpe, spe, mf_rel, mb_rel
    # x should be a 5-tuple: rpe, spe, mf_rel, mb_rel, PMB - updated
    'min-rpe' : (lambda x: x[0] < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x[0] > RPE_HIGH_THRESHOLD),
    'min-spe' : (lambda x: x[1] < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x[1] > SPE_HIGH_THRESHOLD),
    'min-mf-rel' : (lambda x: x[2] < MF_REL_LOW_THRESHOLD),
    'max-mf-rel' : (lambda x: x[2] > MF_REL_HIGH_THRESHOLD),
    'min-mb-rel' : (lambda x: x[3] < MB_REL_LOW_THRESHOLD),
    'max-mb-rel' : (lambda x: x[3] > MB_REL_HIGH_THRESHOLD),
    'min-rpe-min-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['min-spe'](x),
    'max-rpe-max-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['max-spe'](x),
    'min-rpe-max-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['max-spe'](x),
    'max-rpe-min-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['min-spe'](x),
    'random' : lambda x: 0
}


def create_lst(x):
    return [x] * TRIALS_PER_EPISODE

static_action_map = {
    'min-rpe' : create_lst(0),
    'max-rpe' : create_lst(3),
    'min-spe' : create_lst(0),
    'max-spe' : create_lst(1),
    'min-rpe-min-spe' : create_lst(0),
    'max-rpe-max-spe' : create_lst(3),
    'min-rpe-max-spe' : create_lst(1),
    'max-rpe-min-spe' : create_lst(2)
}

def error_to_reward(error, PMB=0 , mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    """Compute reward for the task controller. Based on the input scenario (mode), the reward function is determined from the error_reward_map dict.
        Args:
            error (float list): list with player agent's internal states. Current setting: RPE/SPE/MF-Rel/MB-Rel/PMB
            For the error argument, please check the error_reward_map
            PMB (float): PMB value of player agents. Currently duplicated with error argument.
            mode (string): type of scenario

        Return:
            action (int): action to take by human agent
        """
    if TASK_TYPE == 2019:
        try:
            cmp_func = error_reward_map[mode]
        except KeyError:
            print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
            cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

        return cmp_func(error)
    #elif  TASK_TYPE == 2023 or TASK_TYPE == 2021 or TASK_TYPE == 2020 or TASK_TYPE == 2014:
    else:
        if mode == 'min-rpe':
            reward = (40 - error[0]) * 3
        elif mode == 'max-rpe':
            reward = error[0] * 10
        elif mode == 'min-spe':
            reward = (1 - error[1])*150
        elif mode == 'max-spe':
            reward = error[1]*200
        elif mode == 'min-rpe-min-spe':
            reward = ((40 - error[0]) * 3 + (1 - error[1]) * 150 ) /2
        elif mode == 'max-rpe-max-spe':
            reward = ((error[0]) * 10 + (error[1]) * 100) /2
        elif mode == 'min-rpe-max-spe':
            reward = ((40 - error[0]) * 3 + (error[1]) * 200) /2
        elif mode == 'max-rpe-min-spe':
            reward = ((error[0]) * 10 + (1 - error[1]) * 150) /2
        elif mode == 'min-MF':
            reward = 1 - error[2]
        elif mode == 'max-MF':
            reward = error[2]
        elif mode == 'min-MB':
            reward = 1 - error[3]
        elif mode == 'max-MB':
            reward = error[3]
        elif mode == 'min-MF-min-MB':
            reward = (1 - error[2]) + (1 - error[3])
        elif mode == 'max-MF-max-MB':
            reward = error[2] + error[3]
        elif mode == 'min-MF-max-MB':
            reward = (1 - error[2]) + error[3]
        elif mode == 'max-MF-min-MB':
            reward = error[2] + (1 - error[3])
        elif mode == 'random' :
            reward = 0

        if PMB_CONTROL:
            reward = reward-60*PMB

        return reward  # -60*PMB
#    if cmp_func(error):
#        if CONTROL_REWARD < 0.5 :
#            return CONTROL_REWARD + bias
#        else :
#            return CONTROL_REWARD * ((2-PMB*2)**0.5) + bias
#            #return CONTROL_REWARD*(2-2*PMB) + bias
#    else:
#        return bias

def shuffle_simulation(CONTROL_MODE = 'max-rpe', policy_sbj_indx = 0):
    pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
    rpe_abl = ["20201", "20202", "20203"]
    spe_abl = ["20211", "20212", "20213", "20214", "20215"]
#    if any(x in file_suffix for x in rpe_abl):
#        pol_list = ['min-rpe', 'max-rpe']
#        pol_list_tick = ['r', 'R']
#    elif any(x in file_suffix for x in spe_abl):
#        pol_list = ['min-spe', 'max-spe']
#        pol_list_tick = ['s', 'S']

    if '20240725_delta_control_restore_drop' in file_suffix:
        folderpath = '20240725'  # SPE task with transition probability action change ablation
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

    elif file_suffix == '_2023_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # SPE task with transition probability action change ablation
    elif file_suffix == '_2021_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # previous SPE tasks
    elif file_suffix == '_2020_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # previous RPE tasks
    elif file_suffix == '_2014_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # previous ori tasks
    elif file_suffix in ['_20201_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20202_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20203_20_trials_20240110ab_delta_control_restore_drop_highest']:
        folderpath = '20240110ab'
        #pol_list = ['min-rpe', 'max-rpe']
    elif file_suffix in ['_20211_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20212_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20213_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20214_20_trials_20240110ab_delta_control_restore_drop_highest',
                         '_20215_20_trials_20240110ab_delta_control_restore_drop_highest']:
        folderpath = '20240110ab'
        #pol_list = ['min-spe', 'max-spe']
    elif file_suffix in ['_2020_20_trials_202405222_delta_control_restore_drop_highest',
                         '_2021_20_trials_202405222_delta_control_restore_drop_highest']:
        folderpath = '202405222'
    elif file_suffix == '_2021_20_trials_20240717_delta_control_restore_drop_highest':
        folderpath = '20240717'
        pol_list = ['min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                    'max-MF-min-MB', 'min-MF-max-MB']

    pol = pol_list.index(CONTROL_MODE)
    params = []
    f = open('regdata.csv')
    data = f.readlines()
    for sbj in data:
        params.append(sbj.split(',')[:-1])
    f.close()
    for ii in range(len(params)):
        for jj in range(len(params[ii])):
            params[ii][jj] = float(params[ii][jj])
    TRIALS_PER_EPISODE = 20  # pol_sbj_data.data[pol][0].shape[0]

    # pol_filename = 'history_results/Analysis-Object-'+CONTROL_MODE+'-{0:02d}'.format(policy_sbj_indx)+file_suffix+'.pkl'
    pol_sbj_data = np.zeros((len(pol_list),max_sbj+2,TRIALS_PER_EPISODE))
    pol_filename = 'history_results/' + folderpath + '/optimal_policy' + file_suffix + '.npy'
    tmp = np.squeeze(np.load(pol_filename))
    for ii in range(len(pol_list)):
        pol_sbj_data[ii][0:max_sbj] = tmp[ii]

    print(pol_sbj_data.shape)
    #pol_sbj_data[pol_list.index('min-rpe')][82] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #pol_sbj_data[pol_list.index('max-rpe')][82] = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #pol_sbj_data[pol_list.index('max-rpe')][83] = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]



    PMB_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))
    RPE_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))
    Raw_RPE_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE * 2))
    SPE_shuffle = np.zeros((max_sbj, TOTAL_EPISODES*TRIALS_PER_EPISODE))
    Reward_shuffle=np.zeros((max_sbj,TOTAL_EPISODES*TRIALS_PER_EPISODE))
    Score_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE))
    MBrel_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE))
    MFrel_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE))
    State_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE * 2))
    Action_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE * 2))
    Env_shuffle = np.zeros((max_sbj, TOTAL_EPISODES * TRIALS_PER_EPISODE,4))

    opt_pol = pol_sbj_data[pol][policy_sbj_indx]
    print(opt_pol)

    for affected_sbj_indx in range(max_sbj):
        env = MDP(2, more_control_input=True, legacy_mode=False, task_type=TASK_TYPE,restore_drop_rate=RESTORE_DROP_RATE)
        # initialize human agent one time
        sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=params[affected_sbj_indx][5]) # SARSA model-free learner
        forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=params[affected_sbj_indx][5], disable_cforward=True) # forward model-based learner
        arb     = Arbitrator(AssocRelEstimator(params[affected_sbj_indx][1], env.max_rpe),
                            BayesRelEstimator(thereshold=params[affected_sbj_indx][0]),
                            amp_mb_to_mf=params[affected_sbj_indx][2], amp_mf_to_mb=params[affected_sbj_indx][3], temperature=params[affected_sbj_indx][4], MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
        # register in the communication controller
        env.agent_comm_controller.register('model-based', forward)
        for episode in tqdm(range(TOTAL_EPISODES)):
            if episode > CONTROL_resting:
                env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                          task_type=TASK_TYPE,restore_drop_rate=RESTORE_DROP_RATE)
                #sarsa = sarsa_save
                #forward = forward_save
                #arb = arb_save
                forward.copy(forward_save)
                sarsa.copy(sarsa_save)
                arb.copy(arb_save)
            env.reward_map = env.reward_map_copy.copy()
            env.output_states = env.output_states_copy.copy()
            cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
            cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
            arb.episode_number = episode
            arb.CONTROL_resting = CONTROL_resting
            human_action_list_episode = []
#            env = MDP(2, more_control_input=True, legacy_mode=False, task_type=TASK_TYPE,restore_drop_rate=RESTORE_DROP_RATE)
#            env.agent_comm_controller.register('model-based', forward)
            for trial in range(TRIALS_PER_EPISODE):
                block_indx = trial // int(TRIALS_PER_EPISODE / 4)
                if trial % TRIALS_PER_EPISODE == 0:
                    if episode > CONTROL_resting:
                        env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                                  task_type=TASK_TYPE,restore_drop_rate=RESTORE_DROP_RATE)
                    env.reward_map = env.reward_map_copy.copy()
                    env.output_states = env.output_states_copy.copy()
                if episode <= CONTROL_resting:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE,restore_drop_rate=RESTORE_DROP_RATE)
                env.bwd_idf = -1
                t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = 0
                game_terminate              = False
                human_obs, control_obs_frag = env.reset()
                Env_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial, :] = [env.trans_prob[0],
                                                                                           env.reward_map[1],
                                                                                           env.reward_map[2],
                                                                                           env.reward_map[3]]
                #control_obs                 = np.append(control_obs_frag, [10, 0.5])
                if episode > CONTROL_resting:
                    """control agent choose action"""
                    control_action = int(opt_pol[trial])
                else:
                    if episode > ceil(CONTROL_resting / 4):
                        control_action = random.randrange(0, MDP.NUM_CONTROL_ACTION)
                    else:
                        control_action = 0
                cum_ctrl_act[control_action] += 1
                """control act on environment"""
                if TASK_TYPE == 2019:
                    if control_action == 3:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE in [2021, 20214, 20215]:
                    if control_action == 2:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE in [20211, 20212]:
                    if control_action == 1:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2023:
                    if control_action == 1:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                tmp_bwd_idf = env.bwd_idf
                """control act on environment"""
                if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                    if TASK_TYPE in [20211, 20201] and trial == 0:
                        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, -1])
                    else:
                        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
                if tmp_bwd_idf != env.bwd_idf and env.is_flexible == 0:
                    arb.p_mb = 0.8
                    arb.p_mf = 0.2

                forward.bwd_update(env.bwd_idf, env)
                current_game_step = 0
                #if episode > CONTROL_resting and trial == 0: print(env.reward_map)
                while not game_terminate:
                    """human choose action"""
                    if episode < CONTROL_resting:
                        human_action = randint(0, 1)
                    else:
                        human_action = arb.action(sarsa.get_Q_values(human_obs), forward.get_Q_values(human_obs))
                    #print("human action : ", human_action)
                    Action_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE *2 + trial * 2 + current_game_step] = human_action

                    """human act on environment"""
                    next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                        = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                    State_shuffle[affected_sbj_indx][
                        episode * TRIALS_PER_EPISODE * 2 + trial * 2 + current_game_step] = next_human_obs
                    #print('line 293 : action-' + str(human_action) + ' state-' + str(next_human_obs))
                    """update human agent"""
                    spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
                    if episode <= CONTROL_resting:
                        next_human_action = arb.action(sarsa.get_Q_values(human_obs), forward.get_Q_values(next_human_obs)) # required by models like SARSA
                    else:
                        next_human_action = randint(0,1)
                    if env.is_flexible == 1: #flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else: # specific goal condition human_reward should be normalized to sarsa
                        if human_reward > 0: # if reward is 10, 20, 40
                            # rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                            rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs,
                                                 next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)

                    mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                    t_d_p_mb += d_p_mb
                    t_p_mb   += p_mb
                    t_mf_rel += mf_rel
                    t_mb_rel += mb_rel
                    t_rpe    += abs(rpe)
                    t_spe    += spe
                    t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine

                    """iterators update"""
                    human_obs = next_human_obs
                    if current_game_step == 0:
                        rpe1 = rpe
                        spe1 = spe
                        act1 = human_action
                        stt1 = human_obs
                    else:
                        rpe2 = rpe
                        spe2 = spe
                        act2 = human_action
                        stt2 = human_obs
                    current_game_step += 1

                # calculation after one trial
                d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / 2, [
                t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value
                cum_d_p_mb += d_p_mb
                cum_p_mb   += p_mb
                cum_mf_rel += mf_rel
                cum_mb_rel += mb_rel
                cum_rpe    += rpe
                cum_spe    += spe
                cum_score  += t_score

                """update control agent"""
                t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
                cum_reward += t_reward
                #next_control_obs = np.append(next_control_obs_frag, [rpe, spe])

                #control_obs_extra = [rpe, spe]
                PMB_shuffle[affected_sbj_indx][episode*TRIALS_PER_EPISODE+trial]=p_mb
                RPE_shuffle[affected_sbj_indx][episode*TRIALS_PER_EPISODE+trial]=rpe
                if current_game_step == 0:
                    Raw_RPE_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE * 2 + trial * 2] = rpe1
                else:
                    Raw_RPE_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE * 2 + trial * 2 + 1] = rpe2
                SPE_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial] = spe
                MBrel_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial] = t_mb_rel
                MFrel_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial] = t_mf_rel
                Reward_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial] = t_reward
                Score_shuffle[affected_sbj_indx][episode * TRIALS_PER_EPISODE + trial] = t_score

            if episode == CONTROL_resting - 1:
                arb_save = arb
                sarsa_save = sarsa
                forward_save = forward
    # pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe','max-rpe-min-spe']
    # save_pol_list = ['max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
    # save_file_head = 'history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + save_pol_list[pol_list.index(CONTROL_MODE)]+file_suffix
    save_file_head = 'history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + pol_list[
        pol_list.index(CONTROL_MODE)] + file_suffix
#    save_file_head = 'history_results/20240605/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + pol_list[
#        pol_list.index(CONTROL_MODE)] + file_suffix
#    save_file_head = 'history_results/20240612/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + pol_list[
#        pol_list.index(CONTROL_MODE)] + file_suffix
    np.save(save_file_head+'_PMB.npy',PMB_shuffle)
    np.save(save_file_head+'_RPE.npy',RPE_shuffle)
    np.save(save_file_head + '_RRPE.npy', Raw_RPE_shuffle)
    np.save(save_file_head + '_SPE.npy', SPE_shuffle)
    np.save(save_file_head + '_MBR.npy', MBrel_shuffle)
    np.save(save_file_head + '_MFR.npy', MFrel_shuffle)
    np.save(save_file_head+'_RWD.npy',Reward_shuffle)
    np.save(save_file_head + '_SCR.npy', Score_shuffle)
    np.save(save_file_head + '_STT.npy', State_shuffle)
    np.save(save_file_head + '_ACT.npy', Action_shuffle)
    np.save(save_file_head + '_ENV.npy', Env_shuffle)

    #if policy_sbj_indx >= max_sbj:
    sio.savemat(save_file_head + '_PMB.mat', {'data': PMB_shuffle})
    sio.savemat(save_file_head + '_RPE.mat', {'data': RPE_shuffle})
    sio.savemat(save_file_head + '_RRPE.mat', {'data': Raw_RPE_shuffle})
    sio.savemat(save_file_head + '_SPE.mat', {'data': SPE_shuffle})
    sio.savemat(save_file_head + '_MBR.mat', {'data': MBrel_shuffle})
    sio.savemat(save_file_head + '_MFR.mat', {'data': MFrel_shuffle})
    sio.savemat(save_file_head + '_RWD.mat', {'data': Reward_shuffle})
    sio.savemat(save_file_head + '_SCR.mat', {'data': Score_shuffle})
    sio.savemat(save_file_head + '_STT.mat', {'data': State_shuffle})
    sio.savemat(save_file_head + '_ACT.mat', {'data': Action_shuffle})
    sio.savemat(save_file_head + '_ENV.mat', {'data': Env_shuffle})


if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["policy-sbj=","ctrl-mode=","task-type=",'file-suffix=',"restore-drop-rate="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == "--policy-sbj":
            policy_sbj_indx = int(a)
        elif o == "--ctrl-mode":
            CONTROL_MODE = a
        elif o == "--task-type":
            TASK_TYPE = int(a)
        elif o == "--file-suffix":
            file_suffix = a
            print(file_suffix)
        elif o == "--MF_ONLY":
            MF_ONLY = bool(a)
        elif o == "--MB_ONLY":
            MB_ONLY = bool(a)
        elif o == "--file-suffix":
            FILE_SUFFIX = a
        elif o == "--restore-drop-rate": #In restoring reward actions in controller, sometimes restoring fails
            RESTORE_DROP_RATE = float(a)
        else:
            assert False, "unhandled option"

    shuffle_simulation(CONTROL_MODE = CONTROL_MODE, policy_sbj_indx = policy_sbj_indx)
