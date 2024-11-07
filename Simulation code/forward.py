""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
    modified by Sanghwan Kim <kshwan0227@kaist.ac.kr>
    May 20, 2019
"""
import numpy as np
import common

from collections import defaultdict
USE_CFORWARD = False # True
#try:
# from lib.cforward import cForward
print("C++ FORWARD dynamic library found, set as default backend")
#except ImportError:
#    print("Forward C++ dynamic library not found, only pure Python version availiable")
#    USE_CFORWARD = False

class FORWARD:
    """FORWARD model-based learner

    See the algorithm description from the publication:
    States versus Rewards: Dissociable Neural Prediction Error Signals Underlying Model-Based
    and Model-Free Reinforcement Learning http://www.princeton.edu/~ndaw/gddo10.pdf

    Currently support Discreate observation and action spaces only
    """
    RANDOM_PROBABILITY       = 0.05
    TEMPORAL_DISCOUNT_FACTOR = 1.0
    LEARNING_RATE            = 0.5
    C_SIZE_TRANSITION_PROB   = 2 # C implementation requries knowing the size of transition probability
    NUM_POSSIBLE_NEXT_STATE = 4
    OUTPUT_STATE_ARRAY = [2, 1, 1, 0, 1, 0, 2, 0, 2, 3, 3, 0, 2, 0, 0, 3]
    #OUTPUT_STATE_ARRAY = [1, 2, 2, 3, 2, 3, 1, 3, 1, 0, 0, 3, 1, 3, 3, 0]
    #self.output_states = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
    NINE_STATES_MODE = True
    def __init__(self, observation_space, action_space, state_reward_func, output_offset, reward_map_func,
                 epsilon=RANDOM_PROBABILITY, discount_factor=TEMPORAL_DISCOUNT_FACTOR, learning_rate=LEARNING_RATE,
                 num_possible_next_state = NUM_POSSIBLE_NEXT_STATE, disable_cforward=False,
                 output_state_array = OUTPUT_STATE_ARRAY, nine_states_mode = NINE_STATES_MODE):
        """Args:
            observation_space (gym.spaces.Discrete)
            action_space (gym.spaces.Discrete)
            state_reward_func (closure): a reward map to initialize state-action value dict
            output_offset (int): specify the starting point of terminal reward state
            epsilon (float): thereshold to make a random action
            learning_rate (float)
            discount_factor (float)
        """
        if disable_cforward:
            self.USE_CFORWARD = False
        else:
            self.USE_CFORWARD = USE_CFORWARD
        self.num_states    = observation_space.n
        self.output_offset = output_offset
        self.num_actions   = action_space.n
        self.num_possible_next_state = num_possible_next_state
        self.output_state_array = output_state_array
        self.observe_reward = np.zeros(self.num_states)
        self.nine_states_mode = nine_states_mode
        self.bwd_idf = -1
        if self.USE_CFORWARD:
            self.cforward = cForward(self.num_states, self.num_actions, FORWARD.C_SIZE_TRANSITION_PROB,
                                     self.output_offset, epsilon, learning_rate, discount_factor, num_possible_next_state)
            self.c_reward_array = self.cforward.reward_array
            self.c_reward_map = self.cforward.reward_map
            self.c_state_array = self.cforward.state_array
            self.c_Q_buffer     = self.cforward.Q_buf
            self.env_reset(state_reward_func, reward_map_func)
            self.c_Transition_buffer = self.cforward.Transition_buf
        else: # use pure python
            self.epsilon         = epsilon
            self.discount_factor = discount_factor
            self.learning_rate   = learning_rate
            self.T               = {} # transition matrix
            self.reset(state_reward_func, reward_map_func)
    
    def _Q_fitting(self):
        """Regenerate state-action value dictionary and put it in a closure

        Return:
            python mode: policy_fn (closure)
            C mode: None
        """
        if self.USE_CFORWARD:
            self.cforward.generate_Q()
        else: # use pure python
            self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
            for state in reversed(range(self.num_states)):
                # Do a one-step lookahead to find the best action
                for action in range(self.num_actions):
                    for next_state in reversed(range(self.num_states)):
                        prob, reward = self.T[state][action][next_state]
                        if state >= self.output_offset: # terminal reward states at the bottom of the tree
                            reward = 0
                        best_action_value = np.max(self.Q_fwd[next_state])
                        self.Q_fwd[state][action] += prob * (reward + self.discount_factor * best_action_value)

            # Create a deterministic policy using the optimal value function
            self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)
            return self.policy_fn

    def action(self, state):
        if self.USE_CFORWARD:
            raise NotImplementedError("C mode not implemented, switch to pure python to enable action method")
        return self.policy_fn(state)

    def get_Q_values(self, state):
        """Required by some arbitrition processes

        Note if state >= output_state_offset, then
        python mode: the value will be the higest value in all states times a small transition prob
        C mode: 0
        I am not sure why it is implemented like this in python mode (it is moved from legacy code, 
        so I just keep it), but it should make no much difference.

        Args:
            state (int): a discrete value representing the state
        
        Return:
            Q_values (list): a list of Q values with indices corresponds to specific action
        """
        if self.USE_CFORWARD:
            if state < self.output_offset:
                self.cforward.fill_Q_value_buf(int(state))
                Q_values = []
                # need to copy buffer value to real native list, because some method
                # of list may misbehave like len().
                for i in range(self.num_actions):
                    Q_values.append(self.c_Q_buffer[i])
                return Q_values
            else:
                return np.zeros(self.num_actions)
        else: # use pure python
            return self.Q_fwd[state]

    def optimize(self, state, human_reward, action, next_state, env):
        """Optimize state transition matrix
        
        Args:
        state (int)
            action (int)
            next_state (int)
        
        Returns:
            state_prediction_error (float)
        """
        #print(human_reward)
        #print(env.reward_map)
        if self.nine_states_mode:
            if state >= self.output_offset:
                state = self.output_state_array[state - self.output_offset] + self.output_offset
            if next_state >= self.output_offset:
                next_state = self.output_state_array[next_state - self.output_offset] + self.output_offset
        if self.USE_CFORWARD:
            #print("state : %s, action : %s, next_state : %s" %(int(state), int(action), int(next_state)))
            return self.cforward.optimize(int(state), int(action), int(next_state))
        else: # use pure python
            trans_prob = self.T[state][action]
            for post_state in range(self.num_states):
                prob, reward = trans_prob[post_state]
                if post_state == next_state:
                    if next_state >= self.output_offset :
                        if env.is_flexible == 0:
                            if human_reward != self.observe_reward[next_state] and next_state != self.bwd_idf:
                                self.observe_reward[env.reward_map.index(human_reward) + self.output_offset] = human_reward
                                reward = human_reward
                        else:
                            if human_reward != self.observe_reward[next_state]:
                                self.observe_reward[env.reward_map.index(human_reward) + self.output_offset] = human_reward
                                reward = human_reward
                    spe = 1 - prob
                    trans_prob[post_state] = (prob + self.learning_rate * spe, reward)
                    #print(self.bwd_idf)
                    #print(self.observe_reward)
                    #print(self.bwd_idf)
                else:
                    trans_prob[post_state] = (prob * (1 - self.learning_rate), reward)
            self.T[state][action] = trans_prob
            self._Q_fitting()
            return spe

    def env_reset(self, state_reward_func, reward_map_func):
        """Called by the agent communication controller when environment sends a
        reset signal

        Args:
            state_reward_func (closure): as in constructor
            reward_map_func : function made in mdp.py
        """
        if self.USE_CFORWARD:
            # populate reward array
            for i in range(self.num_possible_next_state):
                self.c_reward_map[i] = int(reward_map_func(i))
            for i in range(self.num_states - self.output_offset):
                self.c_reward_array[i] = int(state_reward_func(i + self.output_offset))
            for i in range(self.num_states - self.output_offset):
                self.c_state_array[i] = int(self.output_state_array[i])
            self.cforward.generate_Q()
        else: # use pure python
            self.reset(state_reward_func, reward_map_func, False)

    def reset(self, state_reward_func, reward_map_func, reset_trans_prob=True):
        #print('reset')
        if self.USE_CFORWARD:
            raise NotImplementedError("full reset method not implemented in C mode")
        self.state_reward_func = state_reward_func
        self.reward_map_func = reward_map_func
        self.observe_reward = np.zeros(self.num_states)
        for state in range(self.num_states):
            if reset_trans_prob:
                self.T[state] = {action: [] for action in range(self.num_actions)}
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    if self.nine_states_mode :
                        if next_state == 0 :
                            self.T[state][action].append(
                                ((0 if reset_trans_prob else self.T[state][action][next_state]), 0))
                        elif next_state < self.output_offset :
                            self.T[state][action].append(
                                ((1. / (self.output_offset - 1) if reset_trans_prob else self.T[state][action][next_state]),
                                 0))
                        else :
                            self.T[state][action].append(
                                ((1. / (self.num_states - self.output_offset) if reset_trans_prob else self.T[state][action][next_state]),
                                 self.reward_map_func(next_state - self.output_offset)))

                    else :
                        if next_state < self.output_offset :
                            self.T[state][action].append(
                                ((1. / self.num_states if reset_trans_prob else self.T[state][action][next_state]),
                                 0))
                        else :
                            self.T[state][action].append(
                                ((1. / self.num_states if reset_trans_prob else self.T[state][action][next_state]),
                                 self.reward_map_func(next_state-self.output_offset)))
                                                  #    self.state_reward_func(next_state)))
            if self.nine_states_mode and state >= self.output_offset:
                state2 = self.output_offset + self.output_state_array.index(state-self.output_offset)
                self.observe_reward[state] = self.state_reward_func(state2)
            else:
                self.observe_reward[state] = self.state_reward_func(state)
        # build state-action value mapping
        self._Q_fitting()

    def get_Transition(self, state, action):
        if self.USE_CFORWARD:
            if state < self.output_offset and action < self.num_actions:
                self.cforward.fill_Transition_buf(int(state), int(action))
                Transition = []
                # need to copy buffer value to real native list, because some method
                # of list may misbehave like len().
                for i in range(self.num_possible_next_state):
                    Transition.append(self.c_Transition_buffer[i])
                return Transition
            else:
                return np.zeros(self.num_possible_next_state)
        else: # use pure python
            return self.T[state][action]

    def bwd_update(self, bwd_idf, env):
        #reward_map=[0,0,0,0,0,40,20,10,0]
        # reward_map = [0,0,0,0,0] + list(reversed(env.reward_map))
        self.bwd_idf = bwd_idf
        reward_map = self.observe_reward
        state_ind_set = (np.int16(np.linspace(1,self.num_possible_next_state,self.num_possible_next_state)).tolist())
        state_ind_set.reverse()
        states_ = [0]
        if self.nine_states_mode == False:
            for sei in range(len(state_ind_set)):
                states_.append(0)
        bwd_output_states = []
        #for ele in env.output_states:
        for ele in self.observe_reward:
            if bwd_idf != -1:
                if int(reward_map[bwd_idf-1]) == int(ele):
                    states_.append(ele)
                    bwd_output_states.append(ele)
                else:
                    states_.append(0)
                    bwd_output_states.append(0)
            else:
                states_.append(ele)
                bwd_output_states.append(ele)

        for cur_st in state_ind_set:
            for cur_act in range(self.num_actions):
                tmp_sum = []
                for j in range((self.num_states)):
                    tmp_sum += self.T[cur_st][cur_act][j][0]*(states_[j]+max(self.Q_fwd[j]))
                self.Q_fwd[cur_st,cur_act]=tmp_sum

        # Create a deterministic policy using the optimal value function
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)

        #env.output_states = bwd_output_states
        self.output_states = bwd_output_states
        self.reward_states = states_

    def copy(self,ori):
        for si in range(self.num_states):
            for ai in range(self.num_actions):
                for si2 in range(self.num_states):
                    self.T[si][ai][si2] = ori.T[si][ai][si2]
        for si in range(self.num_states):
            self.observe_reward[si] = ori.observe_reward[si]
        for si in range(self.num_states):
            for ai in range(self.num_actions):
                self.Q_fwd[si][ai] = ori.Q_fwd[si][ai]
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)

