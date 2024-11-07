""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import torch
import numpy as np
import pandas as pd
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from tqdm import tqdm
from mdp import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS
from common import makedir
import analysis
from main import MODE_LIST

import scipy.io as sio

# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 100
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
DECAY_RATE = 0.5
RESTORE_DROP_RATE = 0.0
turn_off_tqdm = False
CONTROL_resting = 100 #Intial duration for CONTROL agent resting

RESET = False
SAVE_LOG_Q_VALUE = False
MIXED_RANDOM_MODE = False
SAVE_ACTION_PROB = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']


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
    '''
    if TASK_TYPE == 2019:
        try:
            cmp_func = error_reward_map[mode]
        except KeyError:
            print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
            cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

        return cmp_func(error)    
    elif TASK_TYPE == 2020 or TASK_TYPE == 2021:
    '''
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
    else:
        print('wrong mode: ' + mode)

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
    #print([model_free.get_Q_values(human_obs),model_based.get_Q_values(human_obs)])
    return arbitrator.action(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))


def simulation(threshold=BayesRelEstimator.THRESHOLD, estimator_learning_rate=AssocRelEstimator.LEARNING_RATE,
               amp_mb_to_mf=Arbitrator.AMPLITUDE_MB_TO_MF, amp_mf_to_mb=Arbitrator.AMPLITUDE_MF_TO_MB,
               temperature=Arbitrator.SOFTMAX_TEMPERATURE, rl_learning_rate=SARSA.LEARNING_RATE, performance=300, PARAMETER_SET='DEFAULT',
               return_res=False):
    """Simulate single (player agent / task controller) pair with given number of episodes. The parameters for the player agent is fixed during simulation.
        However, the task controller's parameter will be changed (actually, optimized) after each trial.
        During one trial of game, the internal variable of player gents will be collected.
        Then, collected variables will be averaged and turned into the rewards for the task controller.
        The task controller (in here, the Double DQN) will optimize its parameter by reinforcement learning.
        Various variables generated during simulation will be collected and exported to the main.py
        Here's important variable terms :
            episodes : The biggest scope. If 'Reset' variable is activated (Reset = 1), the player agent will be always resumed to the default state (clear Q-value map).
                        Cumulative variables (ex: cum_rpe) is set to 0 at the start of every episode.
                        For the episode 1 ~ 100, the task controller will generate the random actions.
                        This design aims the encouragement of exploration of task controller.
                        Also, the human agents will designed to generate random action during episode 1~99.
                        However, it is encoded in the arbitrator.py, not in the current code.else:
                        The game environment (task structure) will be resumed at the start of every episode, when episode > 100.

            trials : 20 trials = 1 episode in the current setting. 1 trial means one game play, so every trial starts with moving agnet to the initial point of the game.
                    At the start of trial, the task controller changes the task structure, and the storages for the values (ex: t_rpe) are set to 0.
                    At the end of each trial, the task controller will update its parameters.

            game_terminate : The smallest scope. It is boolean value.
                             2 game steps = 1 trial (game trial).
                             At the start of game step, the task player observes the current goal setting and the current state.
                             The player agent choose action with observation, and the next state for that action will be shown.
                             Player agent updates its internal states at the end of game step.

        Args:
            threshold ~ rl_learning_rate : parameters for the player agents
            performance (float) : Fitness of player agent's parameter (not in this simulation)
                                 Currently non-used
            PARAMETER_SET (string) : address for the parameters.
        Return:

    """
    if Session_block:
        MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'random']
    else:
        MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe',
                     'max-rpe-min-spe', 'min-rpe-max-spe', 'min-rpe-PMB', 'max-rpe-PMB', 'min-rpe-max-spe-PMB',
                     'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB', 'max-MF-min-MB', 'min-MF-max-MB',
                     'random']

    CHANGE_MODE_TERM = int(TOTAL_EPISODES/len(RANDOM_MODE_LIST))
    if return_res:
        res_data_df = pd.DataFrame(columns=COLUMNS)
        res_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
    env     = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
    MDP.DECAY_RATE = DECAY_RATE
    analysis.ACTION_COLUMN = ['action_' + str(action_num) for action_num in range(MDP.NUM_CONTROL_ACTION)]
    analysis.COLUMNS = ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'd_p_mb', 'ctrl_reward',
                        'score'] + analysis.ACTION_COLUMN + ['applied_reward']
    if MIXED_RANDOM_MODE:
        random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4 , replace =False) # order among 4 mode is random
        print ('Random Mode Sequence : %s' %random_mode_list)

    # if it is mixed random mode, 'ddpn_loaded' from torch.save(model, filepath) is used instead of 'ddpn'
    ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
                        env.action_space[MDP.CONTROL_AGENT_INDEX],
                        torch.cuda.is_available(), Session_block=Session_block) # use DDQN for control agent

    gData.new_simulation()
    gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])
    control_obs_extra = INIT_CTRL_INPUT

    if Reproduce_BHV:
        saved_policy = np.load('history_results/'+saved_policy_path)
    if not RESET:
        # initialize human agent one time
        sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                        nine_states_mode = env.nine_states_mode) # SARSA model-free learner
        forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION, \
                            output_state_array=  env.output_states_indx, nine_states_mode = env.nine_states_mode)
                        # forward model-based learner
        arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                            BayesRelEstimator(thereshold=threshold),
                            amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
        sarsa_save = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                      nine_states_mode=env.nine_states_mode)  # SARSA model-free learner
        forward_save = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                          env.action_space[MDP.HUMAN_AGENT_INDEX],
                          env.state_reward_func, env.output_states_offset, env.reward_map_func,
                          learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION, \
                          output_state_array=env.output_states_indx, nine_states_mode=env.nine_states_mode)
        # forward model-based learner
        arb_save = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                         BayesRelEstimator(thereshold=threshold),
                         amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature, MB_ONLY=MB_ONLY,
                         MF_ONLY=MF_ONLY)
        # register in the communication controller
        env.agent_comm_controller.register('model-based', forward)

    human_action_list_t = []
    Task_structure = []
    Q_value_forward_t = []
    Q_value_sarsa_t = []
    Q_value_arb_t = []
    Transition_t = []
    #Action_probs_list = np.zeros((MDP.NUM_CONTROL_ACTION,TOTAL_EPISODES * TRIALS_PER_EPISODE))
    Action_probs_list = np.zeros((TOTAL_EPISODES * TRIALS_PER_EPISODE))
    loss_list = np.zeros((TOTAL_EPISODES * TRIALS_PER_EPISODE))
    prev_cum_reward = 0
    Action_probs_list_indx = 0
    if turn_off_tqdm == False:
        episode_list = tqdm(range(TOTAL_EPISODES))
    else:
        episode_list = range(TOTAL_EPISODES)

    for episode in episode_list:
        if episode > CONTROL_resting:
            env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
            #sarsa = sarsa_save
            #forward = forward_save
            #arb = arb_save
            forward.copy(forward_save)
            sarsa.copy(sarsa_save)
            arb.copy(arb_save)

        env.reward_map = env.reward_map_copy.copy()
        env.output_states = env.output_states_copy.copy()
        if RESET:
            # reinitialize human agent every episode
            sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], env, learning_rate=rl_learning_rate,
                            nine_states_mode=env.nine_states_mode) # SARSA model-free learner
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                        env.action_space[MDP.HUMAN_AGENT_INDEX],
                        env.state_reward_func, env.output_states_offset, env.reward_map_func,
                            learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION, \
                            output_state_array=  env.output_states_indx,  nine_states_mode = env.nine_states_mode)
                        # forward model-based learner
            arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                            BayesRelEstimator(thereshold=threshold),
                            amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, MB_ONLY = MB_ONLY, MF_ONLY= MF_ONLY)
            # register in the communication controller
            env.agent_comm_controller.register('model-based', forward)

        if MIXED_RANDOM_MODE and episode % CHANGE_MODE_TERM ==0: # Load ddqn model from exist torch.save every CHANGE_MODE_TERM
            random_mode_index = int(episode/CHANGE_MODE_TERM)
            CONTROL_MODE_temp = random_mode_list[random_mode_index]
            print ('Load DDQN model. Current model : %s Current episode : %s' %(CONTROL_MODE_temp, episode))
            ddqn_loaded = torch.load('ControlRL/' + CONTROL_MODE_temp + '/MLP_OBJ_Subject' + PARAMETER_SET)
            ddqn_loaded.eval() # evaluation mode
            ddqn.eval_Q = ddqn_loaded

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
            if trial%TRIALS_PER_EPISODE == 0:
                if episode > CONTROL_resting:
                    env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                              task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
                env.reward_map = env.reward_map_copy.copy()
                env.output_states = env.output_states_copy.copy()
            if episode <= CONTROL_resting:
                env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE,
                          task_type=TASK_TYPE, restore_drop_rate=RESTORE_DROP_RATE)
            env.bwd_idf = -1
            #if episode > CONTROL_resting : print(env.reward_map)
            #print(env.reward_map)
            t_d_p_mb = t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = rpe1 = rpe2  = 0
            game_terminate              = False
            human_obs, control_obs_frag = env.reset()
            control_obs                 = np.append(control_obs_frag, control_obs_extra)
            if Session_block:
                control_obs = np.append(control_obs, int(str(CONTROL_MODES_LIST)[block_indx]))
            """control agent choose action"""
            if episode > CONTROL_resting:
                if Reproduce_BHV:
                    CONTROL_MODE = CONTROL_MODES_LIST
                    CONTROL_MODE_indx = MODE_LIST.index(CONTROL_MODE)
                    if len(saved_policy.shape) == 1:
                        control_action = int(saved_policy[trial])
                    else:
                        control_action = int(saved_policy[CONTROL_MODE_indx][trial])
                elif trial == 0:
                    action_prob = np.zeros(MDP.NUM_CONTROL_ACTION)
                    action_prob[0] = 1.0
                    control_action = 0
                elif Session_block:
                    if int(str(CONTROL_MODES_LIST)[block_indx]) == 5:
                        control_action = random.randrange(0,MDP.NUM_CONTROL_ACTION)
                elif STATIC_CONTROL_AGENT:
                    control_action = static_action_map[CONTROL_MODE][trial]
                else:
                    action_prob = ddqn.raw_action(control_obs)
                    control_action = ddqn.action(control_obs)
                   # control_action = 0
                cum_ctrl_act[control_action] += 1
                if TASK_TYPE == 2014:
                    if control_action == 2:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE == 2019:
                    if control_action == 3:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE in [2021,20214,20215]:
                    if control_action == 2:
                        if env.is_flexible == 1:
                            arb.p_mb = 0.8
                            arb.p_mf = 0.2
                        else:
                            arb.p_mb = 0.2
                            arb.p_mf = 0.8
                elif TASK_TYPE in [20211,20212]:
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
                    arb.p_mf = 1-arb.p_mb
                elif MF_ONLY:
                    arb.p_mb = 0.0001
                    arb.p_mf = 1 - arb.p_mb
                tmp_bwd_idf = env.bwd_idf
                """control act on environment"""
                if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                    if TASK_TYPE in [20211, 20201] and trial == 0:
                        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, -1])
                    else:
                        _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
                if tmp_bwd_idf != env.bwd_idf and env.is_flexible==0:
                    arb.p_mb = 0.8
                    arb.p_mf = 0.2
                if SAVE_LOG_Q_VALUE:
                    Task_structure.append( np.concatenate((env.reward_map, env.trans_prob, env.output_states), axis=None) )
            else:
                if episode > int(CONTROL_resting/4):
                    control_action = random.randrange(0, MDP.NUM_CONTROL_ACTION)
                    action_prob = np.ones(MDP.NUM_CONTROL_ACTION) / MDP.NUM_CONTROL_ACTION
                    if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
                        if TASK_TYPE in [20211, 20201] and trial == 0:
                            _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, -1])
                        else:
                            _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
                else:
                    control_action = 0
                    action_prob = np.zeros(MDP.NUM_CONTROL_ACTION)
                    action_prob[0] = 1.0

            forward.bwd_update(env.bwd_idf,env)
            #if trial%TRIALS_PER_EPISODE == 0: print(forward.T)
            current_game_step = 0
            while not game_terminate:
                """human choose action"""''
                if episode > CONTROL_resting:
                    human_action = compute_human_action(arb, human_obs, sarsa, forward)
                else:
                    human_action = random.randint(0,1)
                #print("human action : ", human_action)
                if SAVE_LOG_Q_VALUE:
                    human_action_list_episode.append(human_action)

                """human act on environment"""
                next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                """update human agent"""
                spe = forward.optimize(human_obs, human_reward, human_action, next_human_obs, env)
                if episode > CONTROL_resting:
                    next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                else:
                    next_human_action = random.randint(0,1)
                if env.is_flexible == 1: #flexible goal condition
                    rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                else: # specific goal condition human_reward should be normalized to sarsa
                    if human_reward > 0: # if reward is 10, 20, 40
                        #rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else:
                        rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
                #rpe = abs(rpe)
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
            d_p_mb, p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
            t_d_p_mb, t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value

            cum_d_p_mb += d_p_mb
            cum_p_mb   += p_mb
            cum_mf_rel += mf_rel
            cum_mb_rel += mb_rel
            cum_rpe    += rpe
            cum_spe    += spe
            cum_score  += t_score
            """update control agent"""

            if Session_block:
                CONTROL_MODE=MODE_LIST[int(str(CONTROL_MODES_LIST)[block_indx])-1]
                t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
            elif MIXED_RANDOM_MODE:
                t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE_temp)
            else:
                CONTROL_MODE = CONTROL_MODES_LIST
                t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), p_mb, CONTROL_MODE)
            if episode > CONTROL_resting:
                cum_reward += t_reward
                if 'rpe' in CONTROL_MODE or 'spe' in CONTROL_MODE:
                    next_control_obs = np.append(next_control_obs_frag, [rpe, spe])
                elif 'MB' in CONTROL_MODE or 'MF' in CONTROL_MODE:
                    next_control_obs = np.append(next_control_obs_frag, [mf_rel, mb_rel])
                else:
                    print('error in control mode')
                if Session_block:
                    next_control_obs = np.append(next_control_obs, int(str(CONTROL_MODES_LIST)[block_indx]))
                if not MIXED_RANDOM_MODE and (not Reproduce_BHV): # if it is mixed random mode, don't train ddpn anymore. Using ddqn that is already trained before
                    #if trial >= TRIALS_PER_EPISODE/2:
                        # FOR half of trials, let the controller to train players without explicit rewards.
                        # For the next half, let the controller to train players with explicit rewards
                    loss_list[Action_probs_list_indx] = ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
                    real_rwd = t_reward
                    # else:
                      #  real_rwd = 0
                #control_obs_list.append(control_obs)
                #control_action_list.append(control_action)
                #next_control_obs_list.append(next_control_obs)
            else:
                real_rwd = 0
            cum_real_rwd += real_rwd
            #if episode > CONTROL_resting: print(t_reward)

            CONTROL_MODE = CONTROL_MODES_LIST
            if 'rpe' in CONTROL_MODE or 'spe' in CONTROL_MODE:
                control_obs_extra = [rpe, spe]
            elif 'MB' in CONTROL_MODE or 'MF' in CONTROL_MODE:
                control_obs_extra = [mf_rel,mb_rel]
            else:
                print('error in control mode')

            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, d_p_mb, t_reward, t_score] + [control_action] + [rpe1, rpe2] \
                         + [spe1, spe2]
            detail_col = detail_col + env.reward_map + [env.visited_goal_state]
            detail_col = detail_col + [real_rwd]
            detail_col = detail_col + [act1, act2, stt1, stt2, env.trans_prob[0], env.bwd_idf]

            Action_probs_list[Action_probs_list_indx] = control_action
            Action_probs_list_indx += 1

            if not return_res:
                gData.add_detail_res(trial + TRIALS_PER_EPISODE * episode, detail_col)
            else:
                res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col

            if SAVE_LOG_Q_VALUE:
                Q_value_forward = []
                Q_value_sarsa = []
                Q_value_arb = []
                Transition = []
                #print("#############Task_structure##############")
                #print(np.concatenate((env.reward_map, env.trans_prob,env.output_states), axis=None))
                #print("#############forward_Q##############")
                for state in range(5):
                    #print(state ,forward.get_Q_values(state))
                    Q_value_forward += list(forward.get_Q_values(state))
                #print("#############sarsa_Q##############")
                for state in range(5):
                    #print(state ,sarsa.get_Q_values(state))
                    Q_value_sarsa += list(sarsa.get_Q_values(state))
                #print("#############arb_Q##############")
                for state in range(5):
                    #print(state ,arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
                    Q_value_arb += list(arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
                #print("#############Transition##############")
                for state in range(5):
                    for action in range(2):
                        #print(state , action, forward.get_Transition(state, action), sum(forward.get_Transition(state, action)))
                        Transition += list(forward.get_Transition(state, action))

                Q_value_forward_t.append(Q_value_forward)
                Q_value_sarsa_t.append(Q_value_sarsa)
                Q_value_arb_t.append(Q_value_arb)
                Transition_t.append(Transition)

                #print(human_action_list_t, human_action_list_t.shape)
                #print(Task_structure, Task_structure.shape)
                '''print(Transition, len(Transition))
                print(Q_value_forward, len(Q_value_forward))
                print(Q_value_sarsa, len(Q_value_sarsa))
                print(Q_value_arb, len(Q_value_arb))'''

            #if episode > CONTROL_resting : print(forward.Q_fwd)

        #if episode > CONTROL_resting:
        #    for trial in range(TRIALS_PER_EPISODE):
        #        ddqn.optimize(control_obs_list[trial],control_action_list[trial],next_control_obs_list[trial],
        #                      2*cum_real_rwd/TRIALS_PER_EPISODE)

        if episode == CONTROL_resting - 1:
            print('saved')
            #arb_save = arb
            #sarsa_save = sarsa
            #forward_save = forward
            arb_save.copy(arb)
            sarsa_save.copy(sarsa)
            forward_save.copy(forward)

        data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
                                [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_d_p_mb, cum_reward, cum_score]
                                + list(cum_ctrl_act) + [cum_real_rwd]))

        if not return_res:
            gData.add_res(episode, data_col)
        else:
            res_data_df.loc[episode] = data_col
        if SAVE_LOG_Q_VALUE:
            human_action_list_t.append(human_action_list_episode)

        if prev_cum_reward <= cum_reward:
            save_NN = ddqn.eval_Q

        prev_cum_reward = cum_reward

    if SAVE_LOG_Q_VALUE:
        human_action_list_t = np.array(human_action_list_t, dtype=np.int32)
        Task_structure = np.array(Task_structure)
        Transition_t = np.array(Transition_t)
        Q_value_forward_t = np.array(Q_value_forward_t)
        Q_value_sarsa_t = np.array(Q_value_sarsa_t)
        Q_value_arb_t = np.array(Q_value_arb_t)
        '''print(human_action_list_t, human_action_list_t.shape)
        print(Task_structure, Task_structure.shape)
        print(Transition_t, Transition_t.shape)
        print(Q_value_forward_t, Q_value_forward_t.shape )
        print(Q_value_sarsa_t, Q_value_sarsa_t.shape)
        print(Q_value_arb_t, Q_value_arb_t.shape)'''

        gData.add_log_Q_value(human_action_list_t, Task_structure, Transition_t, Q_value_forward_t, Q_value_sarsa_t, Q_value_arb_t )

    if MIXED_RANDOM_MODE:
        #print (list(random_mode_list))
        gData.add_random_mode_sequence(random_mode_list.tolist())

    if SAVE_CTRL_RL and not MIXED_RANDOM_MODE:
        makedir(RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE)
        #torch.save(ddqn.eval_Q.state_dict(), RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save model as dictionary
        torch.save(ddqn.eval_Q, RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save entire model
    if ENABLE_PLOT:
        gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
        gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
        gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    gData.NN = save_NN
    gData.complete_simulation()

    if SAVE_ACTION_PROB:
        save_file_head = 'history_results/20240522' + '/{0:02d}_'.format(NUM_PARAMETER_SET) + CONTROL_MODE + file_suffix
        #sio.savemat(save_file_head + '_CAT.mat', {'data':  np.array(gData.detail[CONTROL_MODE][0]['action'])})
        sio.savemat(save_file_head + '_CAT.mat', {'data': np.array(gData.current_detail_df['action'])})
        sio.savemat(save_file_head + '_LSS.mat', {'data': loss_list})

    if return_res:
        return (res_data_df, res_detail_df)


