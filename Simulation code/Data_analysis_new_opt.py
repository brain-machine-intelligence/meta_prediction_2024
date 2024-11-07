import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import getopt
import sys
from statistics import stdev
from numpy.random import choice
from math import ceil
import torch
import numpy as np
import pandas as pd
import dill as pickle  # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random
import csv
from tqdm import tqdm
from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import analysis
import simulation as sim
from pathlib import Path
import os


def error_to_reward(error,PMB=0,mode='min-rpe'):
    """Compute reward for the task controller. Based on the input scenario (mode), the reward function is determined from the error_reward_map dict.
        Args:
            error (float list): list with player agent's internal states. Current setting: RPE/SPE/MF-Rel/MB-Rel/PMB
            For the error argument, please check the error_reward_map
            PMB (float): PMB value of player agents. Currently duplicated with error argument.
            mode (string): type of scenario

        Return:
            action (int): action to take by human agent
        """
    if mode == 'min-rpe':
        reward = (40 - error[0]) * 3
    elif mode == 'max-rpe':
        reward = error[0] * 10
    elif mode == 'min-spe':
        reward = (1 - error[1]) * 150
    elif mode == 'max-spe':
        reward = error[1] * 200
    elif mode == 'min-rpe-min-spe':
        reward = ((40 - error[0]) * 3 + (1 - error[1]) * 150) / 2
    elif mode == 'max-rpe-max-spe':
        reward = ((error[0]) * 10 + (error[1]) * 100) / 2
    elif mode == 'min-rpe-max-spe':
        reward = ((40 - error[0]) * 3 + (error[1]) * 200) / 2
    elif mode == 'max-rpe-min-spe':
        reward = ((error[0]) * 10 + (1 - error[1]) * 150) / 2
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
    elif mode == 'random':
        reward = 0

    if PMB_CONTROL:
        reward = reward - 60 * PMB

    return reward  # -60*PMB


def compute_human_action(arbitrator, human_obs, model_free, model_based):
    """Compute human action by compute model-free and model-based separately
    then integrate the result by the arbitrator

    Args:
        arbitrator (any callable): arbitrator object
        human_obs (any): valid in human observation space
        model_free (any callable): model-free agent object
        model_based (any callable): model-based agent object

    Return:
        action (int): action to take by human agent
    """
    # print([model_free.get_Q_values(human_obs),model_based.get_Q_values(human_obs)])
    return arbitrator.action(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))


if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["sbj=","mode=","task=","file-suffix=","folderpath=","restore-drop-rate="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == "--sbj":
            sbj_indx = int(a)
        elif o == "--mode":
            control_mode = a
            CONTROL_MODE = control_mode
        elif o == "--task":
            task_type = int(a)
        elif o == "--file-suffix":
            file_suffix = a
        elif o == "--folderpath":
            folderpath = a
        elif o == "--restore-drop-rate": #In restoring reward actions in controller, sometimes restoring fails
            RESTORE_DROP_RATE = float(a)
        else:
            assert False, "unhandled option"

my_file = Path('history_results/' + folderpath)
os.makedirs(my_file, exist_ok=True)

#action_list = ['Nill','R-recover-visited', 'R-recover-unvisited']#['Nil','R-swap', 'R-decay','R-recover']#, 'Uncertainty']
action_list = ['Nil','0.5<->0.9','S<->F','R-recover-visited','R-recover-unvisit']#, 'Uncertainty']
tsne_go = False
NUM_ACT =len(action_list)
pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
if control_mode in ['min-MF','max-MF','min-MB','max-MB','min-MF-min-MB','max-MF-max-MB','max-MF-min-MB','min-MF-max-MB']:
    pol_list = ['min-MF','max-MF','min-MB','max-MB','min-MF-min-MB', 'max-MF-max-MB', 'max-MF-min-MB', 'min-MF-max-MB']
    pol_list_tick = ['minMF(LF)','maxMF(HF)','minMB(LB)','maxMB(HB)','LFLB', 'HFHB', 'HFLB', 'LFHB']
pol_list = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe',
            'min-MF','max-MF','min-MB','max-MB','min-MF-min-MB', 'max-MF-max-MB', 'max-MF-min-MB', 'min-MF-max-MB']
#file_suffix = '_20210616_2019_delta_control'#'_20210331_Q_fix_delta_control'#'_20210329_Q_fix'#'_20210325_RPE' #'_20210304' #_repro_mode202010'
#file_suffix = '_20210601_2021_ep1000_delta_control' #2021 task + 1000 eps
#file_suffix = '_20210601_2021_delta_control' #2021 task + 10000 eps_
#file_suffix = '_20210520_2020_delta_control'
#file_suffix = '20210616_2019_delta_control'
if file_suffix == '_20210520_2020_delta_control':
    pol_list = ['min-rpe', 'max-rpe']
NUM_POL = len(pol_list)
#COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward', 'score']
#COLUMNS = ['rpe','spe','ctrl_reward','score','0','10','20','40','visit', 'applied_reward','rpe1','rpe2','spe1','spe2']
COLUMNS = ['ctrl_reward']
with open('history_results/Analysis-Object-' + pol_list[0] + '-00' + file_suffix + '.pkl','rb') as f:
    data = pickle.load(f)
    NUM_EPISODES, NUM_FULL_FEATS_DATA = data.data[pol_list[0]][0].shape
    NUM_FULL_TRIALS, NUM_FULL_FEATS_DETAIL = data.detail[pol_list[0]][0].shape
    TRIALS_PER_EPISODE = ceil(NUM_FULL_TRIALS / NUM_EPISODES)
max_indx = 82
max_rank = 10
max_rank = 100
test_rep_eps = 100
#TRIALS_PER_EPISODE = 20
#NUM_EPISODES = 1000
full_data = np.zeros((len(COLUMNS),NUM_EPISODES))
full_detail = np.zeros((len(COLUMNS),TRIALS_PER_EPISODE*NUM_EPISODES))
full_opt = np.zeros((len(COLUMNS)))
full_trials_opt = np.zeros((len(COLUMNS),TRIALS_PER_EPISODE))
real_opt = np.zeros((len(COLUMNS)))
full_plot = np.zeros((len(COLUMNS),2))
full_opt_plot = np.zeros((len(COLUMNS),2))
real_opt_plot = np.zeros((len(COLUMNS),2))
full_SBJ_plot = np.zeros((len(COLUMNS),2))
opt_index= np.zeros((max_rank))
opt_pol = np.zeros((max_rank,TRIALS_PER_EPISODE))
final_opt_index = 0
final_opt_pol = np.zeros((TRIALS_PER_EPISODE))
opt_pol_rwd = np.zeros((max_rank,test_rep_eps))
RPE_plot_detail = np.zeros((2))
SPE_plot_detail = np.zeros((2))
PMB_plot_detail = np.zeros((2))
RWD_plot_detail = np.zeros((2))
save_pol = np.zeros((TRIALS_PER_EPISODE))
time_delay = [1.7, 4.06, 6.99]
single_game_time = 8.4
gen_regressor = False
TR = 2.8
full_detail_expanded = np.zeros((ceil(max_indx*NUM_POL*0.8*NUM_EPISODES),ceil(TRIALS_PER_EPISODE *single_game_time/TR)))
pol = pol_list.index(control_mode)
sbj = sbj_indx
print(pol_list[pol])
with open('history_results/Analysis-Object-'+pol_list[pol]+'-{0:02d}'.format(sbj)+file_suffix+'.pkl','rb') as f:
    data = pickle.load(f)
    #data.current_data: eps, data.current_detial : eps*trials

opt_index= data.data[pol_list[pol]][0]['ctrl_reward'].loc[ceil(0.2 * len(data.data[pol_list[pol]][0])):].sort_values().tail(max_rank).index
for pol_rank in range(max_rank):
    for t_indx in range(TRIALS_PER_EPISODE ):
        opt_pol[pol_rank][t_indx] = data.detail[pol_list[pol]][0]['action'].loc[opt_index[pol_rank]*TRIALS_PER_EPISODE -TRIALS_PER_EPISODE +t_indx]


# preset constants
MDP_STAGES = 2
TOTAL_EPISODES = NUM_EPISODES
INIT_CTRL_INPUT = [10, 0.5]
DEFAULT_CONTROL_MODE = 'max-spe'
CTRL_AGENTS_ENABLED = True
RPE_DISCOUNT_FACTOR = 0.003
ACTION_PERIOD = 3
STATIC_CONTROL_AGENT = False
ENABLE_PLOT = True
DISABLE_C_EXTENSION = False
LEGACY_MODE = False
MORE_CONTROL_INPUT = True
SAVE_CTRL_RL = False
PMB_CONTROL = False
TASK_TYPE = task_type
MF_ONLY = False
MB_ONLY = False
Reproduce_BHV = False
saved_policy_path = ''
Session_block = False
mode202010 = False
DECAY_RATE = 0.75
RESTORE_DROP_RATE = 0.0
turn_off_tqdm = True
CONTROL_resting = 100  # Intial duration for CONTROL agent resting
RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']

with open('regdata.csv') as f:
    csv_parser = csv.reader(f)
    param_list = []
    for row in csv_parser:
        param_list.append(tuple(map(float, row[:-1])))
print('Parameter set: ' + str(param_list[sbj]))
[threshold, estimator_learning_rate, amp_mb_to_mf, amp_mf_to_mb, temperature, rl_learning_rate, performance ] =param_list[sbj]

env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
MDP.DECAY_RATE = DECAY_RATE

# initialize human agent one time
sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
              nine_states_mode=env.nine_states_mode)  # SARSA model-free learner
forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                  env.action_space[MDP.HUMAN_AGENT_INDEX],
                  env.state_reward_func, env.output_states_offset, env.reward_map_func,
                  learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION, \
                  output_state_array=env.output_states_indx, nine_states_mode=env.nine_states_mode)
# forward model-based learner
arb = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                 BayesRelEstimator(thereshold=threshold),
                 amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature, MB_ONLY=MB_ONLY,
                 MF_ONLY=MF_ONLY)
# register in the communication controller
env.agent_comm_controller.register('model-based', forward)

prev_cum_reward = 0
if turn_off_tqdm == False:
    episode_list = tqdm(range(CONTROL_resting))
else:
    episode_list = range(CONTROL_resting)

for episode in episode_list:
    env.reward_map = env.reward_map_copy.copy()
    env.output_states = env.output_states_copy.copy()
    arb.episode_number = episode
    arb.CONTROL_resting = CONTROL_resting
    cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = cum_real_rwd = 0
    cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
    human_action_list_episode = []

    control_obs_list = []
    control_action_list = []
    next_control_obs_list = []

    for trial in range(TRIALS_PER_EPISODE):
        block_indx = trial // int(TRIALS_PER_EPISODE / 4)
        if trial % TRIALS_PER_EPISODE == 0:
            env.reward_map = env.reward_map_copy.copy()
            env.output_states = env.output_states_copy.copy()
        env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                  task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
        env.bwd_idf = -1
        # if episode > CONTROL_resting : print(env.reward_map)
        # print(env.reward_map)
        t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2 = 0
        game_terminate = False
        human_obs, control_obs_frag = env.reset()
        if episode > ceil(CONTROL_resting / 4):
            control_action = random.randrange(0, MDP.NUM_CONTROL_ACTION)
        else:
            control_action = 0
        cum_ctrl_act[control_action] += 1
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
        if MB_ONLY:
            arb.p_mb = 0.9999
            arb.p_mf = 1 - arb.p_mb
        elif MF_ONLY:
            arb.p_mb = 0.0001
            arb.p_mf = 1 - arb.p_mb
        tmp_bwd_idf = env.bwd_idf
        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
        if tmp_bwd_idf != env.bwd_idf and env.is_flexible == 0:
            arb.p_mb = 0.8
            arb.p_mf = 0.2
        forward.bwd_update(env.bwd_idf, env)
        current_game_step = 0
        while not game_terminate:
            """human choose action"""
            #human_action = compute_human_action(arb, human_obs, sarsa, forward)
            human_action = random.randint(0,1)
            # print("human action : ", human_action)

            """human act on environment"""
            next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
            """update human agent"""
            spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
            #next_human_action = compute_human_action(arb, next_human_obs, sarsa,
             #                                        forward)  # required by models like SARSA
            next_human_action = random.randint(0, 1)
            if env.is_flexible == 1:  # flexible goal condition
                rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
            else:  # specific goal condition human_reward should be normalized to sarsa
                if human_reward > 0:  # if reward is 10, 20, 40
                    # rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                    rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                else:
                    rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
            # rpe = abs(rpe)
            mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
            t_d_p_mb += d_p_mb
            t_p_mb += p_mb
            t_mf_rel += mf_rel
            t_mb_rel += mb_rel
            t_rpe += abs(rpe)
            t_spe += spe
            t_score += human_reward  # if not the terminal state, human_reward is 0, so simply add here is fine
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
        d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
            t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe]))  # map to average value

        cum_d_p_mb += d_p_mb
        cum_p_mb += p_mb
        cum_mf_rel += mf_rel
        cum_mb_rel += mb_rel
        cum_rpe += rpe
        cum_spe += spe
        cum_score += t_score
        """update control agent"""
        control_action = 0
        t_reward = 0
        real_rwd = t_reward
        cum_real_rwd += real_rwd
        # if episode > CONTROL_resting: print(t_reward)

        control_obs_extra = [rpe, spe]
        detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2] \
                     + [spe1, spe2]
        detail_col = detail_col + env.reward_map + [env.visited_goal_state]
        detail_col = detail_col + [real_rwd]
        detail_col = detail_col + [act1, act2, stt1, stt2, env.trans_prob[0], env.bwd_idf]

    if episode == CONTROL_resting - 1:
        arb_save = arb
        sarsa_save = sarsa
        forward_save = forward
        print('saved')
    data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                        [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score]
                        + list(cum_ctrl_act) + [cum_real_rwd]))
    prev_cum_reward = cum_reward


for pol_indx in range(max_rank):
    print(str(pol_indx))
    saved_policy = opt_pol[pol_indx]
    prev_cum_reward = 0
    if turn_off_tqdm == False:
        episode_list = tqdm(range(test_rep_eps))
    else:
        episode_list = range(test_rep_eps)
    for episode in episode_list:
        env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
        # sarsa = sarsa_save
        # forward = forward_save
        # arb = arb_save
        forward.copy(forward_save)
        sarsa.copy(sarsa_save)
        arb.copy(arb_save)
        env.reward_map = env.reward_map_copy.copy()
        env.output_states = env.output_states_copy.copy()
        arb.episode_number = episode
        arb.CONTROL_resting = CONTROL_resting
        cum_d_p_mb = cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = cum_real_rwd = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
        human_action_list_episode = []

        control_obs_list = []
        control_action_list = []
        next_control_obs_list = []

        for trial in range(TRIALS_PER_EPISODE):
            block_indx = trial // int(TRIALS_PER_EPISODE / 4)
            if trial % TRIALS_PER_EPISODE == 0:
                env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                          task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
                env.reward_map = env.reward_map_copy.copy()
                env.output_states = env.output_states_copy.copy()
            env.bwd_idf = -1
            # if episode > CONTROL_resting : print(env.reward_map)
            # print(env.reward_map)
            t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2 = 0
            game_terminate = False
            human_obs, control_obs_frag = env.reset()
            control_obs = np.append(control_obs_frag, control_obs_extra)
            """control agent choose action"""
            control_action = int(saved_policy[trial])
            # control_action = 0
            cum_ctrl_act[control_action] += 1
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
            if MB_ONLY:
                arb.p_mb = 0.9999
                arb.p_mf = 1 - arb.p_mb
            elif MF_ONLY:
                arb.p_mb = 0.0001
                arb.p_mf = 1 - arb.p_mb
            tmp_bwd_idf = env.bwd_idf
            """control act on environment"""
            _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
            if tmp_bwd_idf != env.bwd_idf and env.is_flexible==0:
                arb.p_mb = 0.8
                arb.p_mf = 0.2
            forward.bwd_update(env.bwd_idf, env)
            current_game_step = 0
            while not game_terminate:
                """human choose action"""
                human_action = compute_human_action(arb, human_obs, sarsa, forward)
                # print("human action : ", human_action)
                if SAVE_LOG_Q_VALUE:
                    human_action_list_episode.append(human_action)

                """human act on environment"""
                next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                """update human agent"""
                spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
                next_human_action = compute_human_action(arb, next_human_obs, sarsa,
                                                         forward)  # required by models like SARSA
                if env.is_flexible == 1:  # flexible goal condition
                    rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                else:  # specific goal condition human_reward should be normalized to sarsa
                    if human_reward > 0:  # if reward is 10, 20, 40
                        rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                    else:
                        rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
                # rpe = abs(rpe)
                mf_rel, mb_rel, p_mb, d_p_mb = arb.add_pe(rpe, spe)
                t_d_p_mb += d_p_mb
                t_p_mb += p_mb
                t_mf_rel += mf_rel
                t_mb_rel += mb_rel
                t_rpe += abs(rpe)
                t_spe += spe

                t_score += human_reward  # if not the terminal state, human_reward is 0, so simply add here is fine
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
            d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
                t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe]))  # map to average value
            cum_d_p_mb += d_p_mb
            cum_p_mb += p_mb
            cum_mf_rel += mf_rel
            cum_mb_rel += mb_rel
            cum_rpe += rpe
            cum_spe += spe
            cum_score += t_score
            """update control agent"""
            t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
            cum_reward += t_reward
            next_control_obs = np.append(next_control_obs_frag, [rpe, spe])

            control_obs_extra = [rpe, spe]
            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2] \
                         + [spe1, spe2]
            detail_col = detail_col + env.reward_map + [env.visited_goal_state]
            detail_col = detail_col + [real_rwd]
            detail_col = detail_col + [act1, act2, stt1, stt2, env.trans_prob[0], env.bwd_idf]

        data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                            [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score]
                            + list(cum_ctrl_act) + [cum_real_rwd]))
        opt_pol_rwd[pol_indx][episode] = cum_reward
        prev_cum_reward = cum_reward

final_opt_rank = np.unravel_index(np.argmax(np.mean(opt_pol_rwd,axis=1)),np.mean(opt_pol_rwd,axis=1).shape)
final_opt_index = opt_index[final_opt_rank]
final_opt_pol = opt_pol[final_opt_rank]

np.save('history_results/' + folderpath + '/final_opt_index_'+control_mode + '_{0:02d}'.format(sbj) + file_suffix + '.npy',final_opt_index)
np.save('history_results/' + folderpath + '/final_opt_pol_'+control_mode + '_{0:02d}'.format(sbj) + file_suffix + '.npy',final_opt_pol)
np.save('history_results/' + folderpath + '/opt_pol_rwd_'+control_mode + '_{0:02d}'.format(sbj) + file_suffix + '.npy',opt_pol_rwd)
temp_mean = np.mean(opt_pol_rwd, axis=1)
temp_std = np.std(opt_pol_rwd,axis = 1)
plt.bar(range(max_rank), temp_mean)
plt.errorbar(range(max_rank), temp_mean, yerr=temp_std)
#plt.xticks(range(max_rank))
plt.title(control_mode + ' ' + str(sbj_indx) + ' top '+str(max_rank)+' pols')
plt.savefig('history_results/' + folderpath + '/'+control_mode + ' ' + str(sbj_indx) + ' top '+str(max_rank)+' pols_plot' + file_suffix + '.png')
plt.clf()

sim.MODE_LIST = pol_list
sim.CONTROL_MODE = control_mode
sim.CONTROL_MODES_LIST = sim.CONTROL_MODE
sim.TOTAL_EPISODES = test_rep_eps+CONTROL_resting
sim.TRIALS_PER_EPISODE = TRIALS_PER_EPISODE
sim.PMB_CONTROL = False
sim.Reproduce_BHV = True
sim.saved_policy_path = folderpath + '/final_opt_pol_' + control_mode + '_{0:02d}'.format(sbj) + file_suffix + '.npy'
sim.DECAY_RATE = 0.75
sim.TASK_TYPE = task_type
sim.ENABLE_PLOT = False
PARAMETER_FILE = 'regdata.csv'
gData.trial_separation = sim.TRIALS_PER_EPISODE
gData.new_mode(sim.CONTROL_MODE)
print('Running mode: ' + sim.CONTROL_MODE)
print(sim.TRIALS_PER_EPISODE)
#            for index in range(NUM_PARAMETER_SET):
print('Parameter set: ' + str(param_list[sbj]))
sim.NUM_PARAMETER_SET = sbj
sim.simulation(*(param_list[sbj]), PARAMETER_SET=str(sbj))
#            gData.generate_summary(sim.CONTROL_MODE)
gData.save_mode(sim.CONTROL_MODE)
pkl_file_name = folderpath + '/Analysis-Object'
pkl_file_name += '-'
pkl_file_name += control_mode
pkl_file_name += '-'
pkl_file_name += '{0:02d}'.format(sbj)
print(pkl_file_name)
with open(gData.file_name(pkl_file_name) + file_suffix + '_highest.pkl', 'wb') as f:
    pickle.dump(gData, f, pickle.HIGHEST_PROTOCOL)

'''
for feat_indx in range(len(COLUMNS)):
    # if feat_indx == 0: full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
    full_data[feat_indx][pol][sbj] = data.data[pol_list[pol]][0][COLUMNS[feat_indx]]
    full_detail[feat_indx][pol][sbj] = data.detail[pol_list[pol]][0][COLUMNS[feat_indx]]
    opt_index[pol][sbj] = data.data[pol_list[pol]][0]['ctrl_reward'].loc[
                          ceil(0.2 * len(data.data[pol_list[pol]][0])):].idxmax()
    #print(data.data[pol_list[pol]][0]['ctrl_reward'][opt_index[pol][sbj]])
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
    full_plot[feat_indx][pol][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*TRIALS_PER_EPISODE*NUM_EPISODES):])/max_indx
    full_SBJ_plot[feat_indx][sbj][0] += np.mean(full_detail[feat_indx][pol][sbj][ceil(0.2*TRIALS_PER_EPISODE*NUM_EPISODES):])/4
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

for sbj in range(max_indx):
    tmp = np.transpose(full_opt[feat_indx])
    full_SBJ_plot[feat_indx][sbj][1] = stdev(tmp[sbj])/np.sqrt(len(pol_list))

np.save('history_results/optimal_policy'+file_suffix+'.npy',opt_pol)
np.save('history_results/feat'+file_suffix+'.npy',full_detail)
print(full_opt.shape)
np.save('history_results/feat_full_opt'+file_suffix+'.npy',full_opt)
np.save('history_results/feat_full_opt_full_trials'+file_suffix+'.npy',full_trials_opt)
np.save('history_results/optimal_policy_index'+file_suffix+'.npy',opt_index)
#if gen_regressor == True:
#   scipy.io.savemat('history_results/RPE_regressor'+file_suffix+'.mat',{'RPE': full_detail_expanded})

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
    plt.savefig('history_results/'+COLUMNS[feat_indx]+'_plot' + file_suffix + '.png')
    plt.clf()


for feat_indx in range(len(COLUMNS)):
    for mode_idf in range(len(pol_list)):
        #plt.plot(np.transpose(full_detail[feat_indx][mode_idf]))
        #plt.savefig('history_results/'+COLUMNS[feat_indx]+'_'+pol_list[mode_idf]+'_full_plot' + file_suffix + '.png')
        #plt.clf()
        plt.plot(np.transpose(full_data[feat_indx][mode_idf]))
        plt.savefig(
            'history_results/' + COLUMNS[feat_indx] + '_' + pol_list[mode_idf] + '_abstract_plot' + file_suffix + '.png')
        plt.clf()

for pol in range(NUM_POL):
    print(pol_list[pol])
    pol_acts = np.zeros((NUM_ACT, TRIALS_PER_EPISODE ))
    for ii in range(82):
        for t_indx in range(TRIALS_PER_EPISODE ):
            pol_acts[int(opt_pol[pol][ii][t_indx][0])][t_indx] += 1

    pol_acts /= 82

    for ii in range(len(pol_acts)):
        plt.plot(pol_acts[ii], label=ii)
    plt.legend(action_list, loc=5)
    plt.ylabel('Action Frequency')
    plt.xlabel('Episode')
    plt.title('Action frequency in the '+pol_list[pol]+' optimal sequences')
    plt.ylim((0, 1))
    plt.savefig('history_results/Action_frequency_'+pol_list[pol] + file_suffix +'_opt.png')
    plt.clf()
'''
    
    
