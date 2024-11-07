""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import random
import numpy as np
import gym
import random
from numpy.random import choice
from gym import spaces
from common import AgentCommController

class MDP(gym.Env):
    """Markov Decision Process env class, inherited from gym.Env
    Although there is not much code in gym.Env, it is just
    showing we want to support general gym API, to be able
    to easily run different environment with existing code in the future

    The MDP implemented here support arbitrary stages and arbitrary
    possible actions at any state, but each state share the same number of
    possible actions. So the decision tree is an n-tree
    """

    """MDP constants
    
    Access of observation and action space should refer
    to these indices
    """
    HUMAN_AGENT_INDEX   = 0
    CONTROL_AGENT_INDEX = 1

    LEGACY_MODE           = False
    STAGES                = 2
    TRANSITON_PROBABILITY = [1.0, 0.0]
    #[0.5, 0.5] #[0.9, 0.1]
    NUM_ACTIONS           = 2
    POSSIBLE_OUTPUTS      = [0, 10, 20, 40]
    BIAS_TOGGLE_INCREMENT = 40
    TASK_TYPE = 2020
    DECAY_RATE = 0.75
    NINE_STATES_MODE      = True
    reader_mode = False
    num_tags = 4
    """2019: old version, 2020: reward decay version"""
    restore_drop_rate = 0
    """Control Agent Action Space
    0 - doing nothing
    1 - set stochastic: apply uniform distribution of transition probability
    2 - set deterministic: set transition probability to original one
    3 - randomize human reward: reset reward shape
    4 - reset: randomize human reward and set deterministic transition probability
    5 - reset2: randomize human reward and set stochastic transition probability
    """
    NUM_CONTROL_ACTION = 5
    def __init__(self, stages=STAGES, trans_prob=TRANSITON_PROBABILITY, num_actions=NUM_ACTIONS,
                 outputs=POSSIBLE_OUTPUTS, more_control_input=True, legacy_mode=LEGACY_MODE, task_type=TASK_TYPE,
                 decay_rate = DECAY_RATE, reader_mode = False, num_tags = 4, restore_drop_rate = 0):
        """
        Args:
            stages (int): stages of the MDP
            trans_prob (list): an array specifying the probability of transitions
            num_actions (int): number of actions possible to take at non-leaf state
                by player. Note total number of possible actions should be multiplied
                by the size of trans_prob
            outputs (list): an array specifying possible outputs
            more_control_input (bool): more element in control observation
            legacy_mode (bool): if use legacy implementation of MDP
        """
        # environment global variables
        self.nine_states_mode = self.NINE_STATES_MODE
        self.task_type = task_type
        if self.task_type == 2014:
            MDP.NUM_CONTROL_ACTION = 3
            trans_prob = [0.9, 0.1]
        elif self.task_type == 2019:
            MDP.NUM_CONTROL_ACTION = 4
            trans_prob = [0.9, 0.1]
        elif self.task_type == 2020:
            MDP.NUM_CONTROL_ACTION = 3
            trans_prob = [0.5, 0.5]
        elif self.task_type in [20201,20202,20203]:
            MDP.NUM_CONTROL_ACTION = 2
            trans_prob = [0.5, 0.5]
        elif self.task_type == 2021:
            MDP.NUM_CONTROL_ACTION = 5
            trans_prob = [0.9, 0.1]
        elif self.task_type in [20211,20212,20213,20214,20215]:
            MDP.NUM_CONTROL_ACTION = 4
            trans_prob = [0.9, 0.1]
        elif self.task_type == 2023:
            MDP.NUM_CONTROL_ACTION = 4
            trans_prob = [0.9, 0.1]
        self.stages            = stages
        self.human_state       = 0 # start from zero
        self.legacy_mode       = legacy_mode
        if self.legacy_mode:
            self.max_rpe       = outputs[-1] + MDP.BIAS_TOGGLE_INCREMENT
            self.toggle_bit    = 0
        else:
            self.max_rpe       = outputs[-1]
        self.reward_map = [0, 10, 20, 40]  # this is corressponding to state 5, 6, 7, 8
        self.reward_map_copy = self.reward_map.copy()
        self.reward_prob_map = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]  # [[Gaining %, Lossing %]]

        # human agent variables
        self.action_space      = [spaces.Discrete(num_actions)] # human agent action space
        self.trans_prob        = trans_prob
        self.possible_actions  = len(self.trans_prob) * num_actions
        self.outputs           = outputs # type of outputs
        self.num_output_states = pow(self.possible_actions, self.stages)
        self.output_states = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]#choice(outputs, self.num_output_states)
        self.output_states_indx = []
        for ii in range(len(self.output_states)):
            self.output_states_indx.append(self.reward_map.index(self.output_states[ii]))
        self.output_states_copy = self.output_states.copy()
        self.reward_states_prob_map =[]
        for indx in range(len(self.output_states)):
            self.reward_states_prob_map.append([1.0,0.0])
        self.output_states_offset = int((pow(self.possible_actions, self.stages) - 1)
            / (self.possible_actions - 1)) # geometric series summation
        if self.nine_states_mode:
            self.num_states = 9
            self.output_states_offset = 5
        elif self.legacy_mode:
            self.num_states    = self.output_states_offset + len(self.outputs)
        else:
            self.num_states    = self.output_states_offset + self.num_output_states
        self.observation_space = [spaces.Discrete(self.num_states)] # human agent can see states only
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        self.remember_output_states = self.output_states.copy()
        self.remember_reward_map = self.reward_map.copy()
        self.is_flexible = 1 #  1 if flexible goal condition and 0 if spicific goal condition
        self.visited_goal_state = -1 #


        # control agent variables
        self.more_control_input = more_control_input
        self.action_space.append(spaces.Discrete(MDP.NUM_CONTROL_ACTION)) # control agent action space
        if more_control_input:
            if legacy_mode:
                output_structure = spaces.Discrete(1) # one toggle bit
            else:
                output_structure = spaces.Discrete(self.num_output_states) # output states
            if reader_mode:
                self.observation_space.append(spaces.Tuple((
                    output_structure,  # depends on if it is legacy mode or
                    spaces.Box(low=0, high=1, shape=(num_actions,), dtype=float),  # transition probability
                    spaces.Box(low=0, high=1, shape=(num_tags,), dtype=float))))  # tag probability
            else:
                self.observation_space.append(spaces.Tuple((
                    output_structure, # depends on if it is legacy mode or
                    spaces.Box(low=0, high=1, shape=(num_actions,), dtype=float), # transition probability
                    spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                    spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe
        else:
            self.observation_space.append(spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe
        self.decay_rate = decay_rate
        self.restore_drop_rate = restore_drop_rate

        # for reset reference
        self.trans_prob_reset = trans_prob

        # agent communication controller
        self.agent_comm_controller = AgentCommController()

        # Previous goal type for backward learning. -1=fle, 6,7,8=spe 0=starting(no data)
        self.bwd_idf = 0


    def _make_state_reward_func(self):
        """
        rwd_prob = lambda s: self.reward_states_prob_map[s - self.output_states_offset] \
               if s >= self.output_states_offset else 0
        rwd_prob = rwd_prob > random.random()
        rwd_amount = lambda s: self.output_states[s - self.output_states_offset] \
               if s >= self.output_states_offset else 0
        """
        return lambda s: (self.reward_states_prob_map[s - self.output_states_offset][0]>random.random())* \
                         (self.output_states[s - self.output_states_offset]) \
            if s >= self.output_states_offset else 0

    def _make_reward_map_func(self):
        """
        rwd_prob = lambda s: self.reward_map[s] \
            if s s < len(self.reward_map)  else 0
        rwd_prob = rwd_prob > random.random()
        rwd_amount = lambda s: self.reward_map[s] \
               if s < len(self.reward_map) else 0
        """
        return lambda s: (self.reward_prob_map[s][0] >random.random())*self.reward_map[s] \
            if s < len(self.reward_map) else 0

    def _make_control_observation(self):
        if self.more_control_input:
            if self.legacy_mode:
                return np.concatenate([[self.toggle_bit], np.array(self.trans_prob)])
            else:
                return np.concatenate([self.output_states, np.array(self.trans_prob)])
        else:
            return []

    def step(self, action):
        """"Take one step in the environment
        
        Args:
            action ([int, action]): a two element tuple, first sepcify which agent
            second is the action valid in that agent's action space

        Return (human):
            human_obs (int): an integer represent human agent's observation, which is
            equivalent to 'state' in this environment
            human_reward (float): reward received at the end of the game
            done (boolean): if the game termiate
            control_obs_frag (numpy.array): fragment of control observation, need to append reward
        Return (control):
            None, None, None, None: just match the arity
        """

        if action[0] == MDP.HUMAN_AGENT_INDEX:
            """ Human action
            Calculate the index of the n-tree node, start from 0
            Each node has possible_actions childs, the calculation is a little tricky though.
            Output_states_offset is the max index of internal nodes + 1
            Greater or equal to output_states_offset means we need to get reward from output_states
            """
            state = self.human_state * self.possible_actions + \
                    choice(range(action[1] * len(self.trans_prob) + 1, (action[1] + 1) * len(self.trans_prob) + 1),
                           1, True, self.trans_prob)[0]
            #print(self.trans_prob)
            #print(str(range(self.human_state * self.possible_actions + action[1] * len(self.trans_prob) + 1, self.human_state * self.possible_actions + (action[1] + 1) * len(self.trans_prob) + 1)))
            reward = self.state_reward_func(state)
            if self.nine_states_mode == 1 and state > 4:
                state = self.output_states_indx[state-5]+5
            self.human_state = state
            if state < self.output_states_offset:
                done = False
            else:
                done = True
                self.visited_goal_state = state
                if self.legacy_mode:
                    self.human_state = self.output_states_offset + self.outputs.index(reward)
                    reward += MDP.BIAS_TOGGLE_INCREMENT if self.toggle_bit else 0
            return self.human_state, reward, done, self._make_control_observation()
        elif action[0] == MDP.CONTROL_AGENT_INDEX:
            """ Control action
            Integrate functional and object oriented programming techniques
            to create this pythonic, compact code, similar to switch in other language
            """
            #if self.task_type == 2020 or self.task_type == 2021:
            if self.task_type in [2020, 2021,20201, 20202,20203,20211,20212,20213,20214,20215]:
            # First decrease visited reward
                self.decay_rwd()
                #if self.task_type == 2021 and random.random() < 0.5: self.restore_rwd()
                #self.decay_prob()

            if action[1] != -1:
                if self.task_type == 2014:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(), # uniform trans_prob                 s
                     lambda env: env._specific_flexible_switch()
                     ][action[1]](self)
                elif self.task_type == 2019:
                    [lambda env: env, # do nothing
                     lambda env: env._set_stochastic_trans_prob(), # uniform trans_prob
                     lambda env: setattr(env, 'trans_prob', env.trans_prob_reset), # reset trans prob to deterministic
                     #lambda env: env._set_opposite_trans_prob(),
                     lambda env: env._specific_flexible_switch()
                     #lambda env: env._output_swap(),
                     #lambda env: env._reward_shift()
                     #env._output_reset()   # reset reward shape
                     #lambda env: env._output_reset_with_deterministic_trans_prob(), # reset
                     #lambda env: env._output_reset_with_stochastic_trans_prob() # reset2
                     #lambda env: env._shuffle_reward(),25
                     #lambda env: env._reward_shift_right()# lambda env: env._reward_shift_left()
                     #lambda env: env.decay_prob(),
                     #lambda env: env.restore_prob(),
                     #lambda env: env.decay_rwd(),
                     #lambda env: env.restore_rwd(),
                     #lambda env: env.shift_high_low_uncertainty(), # trans_prob [0.5 0.5]->[0.9 0.1] / [0.9 0.1]->[0.5 0.5]
                     #lambda env: env.reward_contrast(), # decay_rwd + restore_rwd
                     #lambda env: env.reward_prob_contrast # decay_prob + restore_prob
                    ][action[1]](self)
                elif self.task_type == 2020:
                    [lambda env: env,  # do nothing,
                     lambda env: env.restore_visited_rwd(),
                     lambda env: env.restore_rwd()
                     #lambda env: env.shift_high_low_uncertainty() # trans_prob [0.5 0.5]->[0.9 0.1] / [0.9 0.1]->[0.5 0.5]
                     ][action[1]](self)
                elif self.task_type == 20201:
                    [lambda env: env.restore_visited_rwd(),
                     lambda env: env.restore_rwd()
                     ][action[1]](self)
                elif self.task_type == 20202:
                    [lambda env: env,  # do  nothing,
                     lambda env: env.restore_rwd()
                     ][action[1]](self)
                elif self.task_type == 20203:
                    [lambda env: env,  # do nothing,
                     lambda env: env.restore_visited_rwd(),
                     ][action[1]](self)
                elif self.task_type == 2021:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(), # swap trans_prob between [0.5 0.5] and [0.9 0.1]
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                elif self.task_type == 20211:
                    [lambda env: env.shift_high_low_uncertainty(), # swap trans_prob between [0.5 0.5] and [0.9 0.1]
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                elif self.task_type == 20212:
                    [lambda env: env, # do nothing
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                elif self.task_type == 20213:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(), # swap trans_prob between [0.5 0.5] and [0.9 0.1]
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                elif self.task_type == 20214:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(), # swap trans_prob between [0.5 0.5] and [0.9 0.1]
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                elif self.task_type == 20215:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(), # swap trans_prob between [0.5 0.5] and [0.9 0.1]
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                    ][action[1]](self)
                elif self.task_type == 2023:
                    [lambda env: env, # do nothing
                     lambda env: env.shift_high_low_uncertainty(),  # uniform trans_prob
                     lambda env: env._specific_flexible_switch(), # swap goal condition between flexible and specific
                     lambda env: env.restore_visited_rwd(), # restore_visited_rwd()
                     lambda env: env.restore_rwd() # restore_unvisited_rwd()
                    ][action[1]](self)
                #print(self.output_states)

            if self.is_flexible == 0:  # specific goal condition
                #self.remember_output_states = self.output_states.copy()
                #self.remember_reward_map = self.reward_map.copy()
                #specific_goal = random.choice([10, 20, 40])
                specific_goal = random.choice(self.remember_reward_map[1:])
                #print([self.reward_map_copy, specific_goal, self.remember_reward_map, self.reward_map, '1'])
                self.reward_map = list(map((lambda x: self.remember_reward_map[self.remember_reward_map.index(specific_goal)] if x == specific_goal else
                0), self.remember_reward_map))
                self.output_states = list(map((lambda x: self.remember_output_states[self.remember_output_states.index(specific_goal)] if x == specific_goal else
                0), self.remember_output_states))
                #print([self.reward_map_copy, specific_goal, self.remember_reward_map,self.reward_map,'2'])

                # refresh the closure as well
                self.state_reward_func = self._make_state_reward_func()
                self.reward_map_func = self._make_reward_map_func()
                # reset human agent
                self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

            # save previous goal type for the next trial

            if self.is_flexible == 1: # flexible goal condition
                self.bwd_idf = -1

            else:
                if specific_goal == 10:
                    self.bwd_idf = 6
                elif specific_goal == 20:
                    self.bwd_idf = 7
                elif specific_goal == 40:
                    self.bwd_idf = 8

            return None, None, None, None
        else:
            raise ValueError
###################################my own code##############################
    def _shuffle_reward(self):
        self.output_states = list(map((lambda x : 5 if x == self.reward_map[0] else
                                                  6 if x == self.reward_map[1] else
                                                  7 if x == self.reward_map[2] else
                                                  8), self.output_states))
        random.shuffle(self.reward_map)
        #change state number 5-8 to reward according to shuffled reward_map
        self.output_states = list(map((lambda x : self.reward_map[0] if x == 5 else
                                                  self.reward_map[1] if x == 6 else
                                                  self.reward_map[2] if x == 7 else
                                                  self.reward_map[3]), self.output_states))

    def _reward_shift(self):
        determinator = random.randint(0,1)
        if determinator:
            self.output_states = list(map((lambda x: 0 if x == 10 else
                                                     10 if x == 20 else
                                                     20 if x == 40 else
                                                     40), self.output_states))
            self.reward_map = list(map((lambda x: 0 if x == 10 else
                                                  10 if x == 20 else
                                                  20 if x == 40 else
                                                  40), self.reward_map))
        else:
            self.output_states = list(map((lambda x: 0 if x == 40 else
                                                     10 if x == 0 else
                                                     20 if x == 10 else
                                                     40), self.output_states))
            self.reward_map = list(map((lambda x: 0 if x == 40 else
                                                  10 if x == 0 else
                                                  20 if x == 10 else
                                                  40), self.reward_map))
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()

    def _reward_shift_right(self):
        self.output_states = list(map((lambda x: 0 if x == 10 else
        10 if x == 20 else
        20 if x == 40 else
        40), self.output_states))
        self.reward_map = list(map((lambda x: 0 if x == 10 else
        10 if x == 20 else
        20 if x == 40 else
        40), self.reward_map))
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()

    def _reward_shift_left(self):
        self.output_states = list(map((lambda x: 0 if x == 40 else
        10 if x == 0 else
        20 if x == 10 else
        40), self.output_states))
        self.reward_map = list(map((lambda x: 0 if x == 40 else
        10 if x == 0 else
        20 if x == 10 else
        40), self.reward_map))
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()


 #   def _shuffle_add_reward(self):



#############################################################################
    def _set_stochastic_trans_prob(self):
        self.trans_prob = [1./len(self.trans_prob) for i in range(len(self.trans_prob))]


    def _set_opposite_trans_prob(self):
        self.trans_prob = [0.1, 0.9]


    def _output_swap(self):
        self.output_states = list(map((lambda x: 10 if x == 20 else
                                                 0 if x == 40 else
                                                 40 if x == 0 else
                                                 20), self.output_states))
        self.reward_map = list(map((lambda x: 10 if x == 20 else
                                                 0 if x == 40 else
                                                 40 if x == 0 else
                                                 20), self.reward_map))
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

    def _output_average_with_stochastic_trans_prob(self):
        self.output_states = [0.9 * (x - 20) + 20 for x in self.output_states]
        self.state_reward_func = self._make_state_reward_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)
        self._set_stochastic_trans_prob()

    def _output_reset_with_stochastic_trans_prob(self):
        self._output_reset()
        self._set_stochastic_trans_prob()

    def _output_reset_with_deterministic_trans_prob(self):
        self._output_reset()
        self.trans_prob = self.trans_prob_reset

    def _output_reset(self):
        """Reset parameters, used as an action in control agent space
        """
        #change output_states to state number 5-8
        self.output_states = list(map((lambda x : 5 if x == self.reward_map[0] else 
                                                  6 if x == self.reward_map[1] else 
                                                  7 if x == self.reward_map[2] else 
                                                  8), self.output_states))  
        random.shuffle(self.reward_map)
        #change state number 5-8 to reward according to shuffled reward_map
        self.output_states = list(map((lambda x : self.reward_map[0] if x == 5 else 
                                                  self.reward_map[1] if x == 6 else 
                                                  self.reward_map[2] if x == 7 else 
                                                  self.reward_map[3]), self.output_states))      
        #self.output_states = choice(self.outputs, self.num_output_states)


        # refresh the closure as well
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        # reset human agent
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

    def _specific_flexible_switch(self):
        """Specific goal condition <-> Flexible goal condition(10 or 20 or 40 is randomly chosen)"""
        self.is_flexible = (self.is_flexible + 1) % 2 #  1 if flexible goal condition and 0 if spicific goal condition
        if self.is_flexible == 0: # speicific goal condition
            self.remember_output_states = self.output_states.copy()
            self.remember_reward_map = self.reward_map.copy()
            # specific_goal = random.choice([10, 20, 40])
            specific_goal = random.choice(self.remember_reward_map[1:])
            self.reward_map = list(map((lambda x: self.remember_reward_map[
                self.remember_reward_map.index(specific_goal)] if x == specific_goal else
            0), self.remember_reward_map))
            self.output_states = list(map((lambda x: self.remember_output_states[
                self.remember_output_states.index(specific_goal)] if x == specific_goal else
            0), self.remember_output_states))
            # self.output_states = list(map((lambda x : specific_goal if x == specific_goal else
            #                                           0), self.output_states))
            # self.reward_map = list(map((lambda x : specific_goal if x == specific_goal else
            #                                           0), self.reward_map))
            # refresh the closure as well
            self.state_reward_func = self._make_state_reward_func()
            self.reward_map_func = self._make_reward_map_func()
            # reset human agent
            self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)
        else: # flexible goal condition
            self.output_states = self.remember_output_states.copy()
            self.reward_map = self.remember_reward_map.copy()
            # refresh the closure as well
            self.state_reward_func = self._make_state_reward_func()
            self.reward_map_func = self._make_reward_map_func()
            # reset human agent
            self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

        
    def reset(self):
        """Reset the environment before game start or after game terminates

        Return:
            human_obs (int): human agent observation
            control_obs_frag (numpy.array): control agent observation fragment, see step
        """
        self.human_state = 0

        return self.human_state, self._make_control_observation()

    def decay_prob(self):
        """
        Discounts the previously visited state's reward probability.
        """
        if self.visited_goal_state > 0:
            if self.nine_states_mode == 1:
                random_number = random.random()
                reward_map_indx = self.visited_goal_state - self.output_states_offset
                self.reward_prob_map[reward_map_indx][0] *= max(0, self.decay_rate - 0.1 + 0.2 * random_number)
                self.reward_prob_map[reward_map_indx][1] = 1 - self.reward_prob_map[reward_map_indx][0]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[self.visited_goal_state - self.output_states_offset]:
                        self.reward_states_prob_map[indx][0] *= max(0, self.decay_rate - 0.1 + 0.2 * random_number)
                        self.reward_states_prob_map[indx][1] = 1 - self.reward_states_prob_map[indx][0]
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                reward_map_indx = self.reward_map_copy.index(self.output_states_copy[self.visited_goal_state - self.output_states_offset])
                self.reward_prob_map[reward_map_indx][0] *= max(0, self.decay_rate - 0.1 + 0.2*random_number)
                self.reward_prob_map[reward_map_indx][1] = 1 - self.reward_prob_map[reward_map_indx][0]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.output_states_copy[self.visited_goal_state - self.output_states_offset]:
                        self.reward_states_prob_map[indx][0] *= max(0, self.decay_rate - 0.1 + 0.2*random_number)
                        self.reward_states_prob_map[indx][1] = 1 - self.reward_states_prob_map[indx][0]
                self.state_reward_func = self._make_state_reward_func()



    def decay_rwd(self):
        """
        Discounts the previously visited state's reward value.
        """
        if self.visited_goal_state > 0:
            if self.nine_states_mode == 1:
                random_number = random.random()
                reward_map_indx = self.visited_goal_state - self.output_states_offset
                self.reward_map[reward_map_indx] *= max(0.1, self.decay_rate - 0.1 + 0.2*random_number)
                if self.reward_map[reward_map_indx] < 1 : self.reward_map[reward_map_indx] = 1
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0.1, self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] < 1 : self.output_states[indx] = 1
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                reward_map_indx = self.reward_map_copy.index(
                    self.output_states_copy[self.visited_goal_state - self.output_states_offset])
                self.reward_map[reward_map_indx] *= max(0.1, self.decay_rate - 0.1 + 0.2*random_number)
                if self.reward_map[reward_map_indx] < 1 : self.reward_map[reward_map_indx] = 1
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0.1, self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] < 1 : self.output_states[indx] = 1
                self.state_reward_func = self._make_state_reward_func()


    def restore_prob(self):
        """
            restore decayed reward prob of unvisited states
        """
        if self.visited_goal_state > 0:
            if self.nine_states_mode == 1:
                random_number = random.random()
                unvisited_state_list = []
                for any_state in range(len(self.reward_map_copy)):
                    if self.reward_map_copy[any_state] != self.reward_map_copy[self.visited_goal_state - self.output_states_offset]:
                        unvisited_state_list.append(any_state)
                for unvisited_state in unvisited_state_list:
                    self.reward_prob_map[unvisited_state][0] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                    if self.reward_prob_map[unvisited_state][0] > 1.0:
                        self.reward_prob_map[unvisited_state][0] = 1.0
                    self.reward_prob_map[unvisited_state][1] = 1.0 - self.reward_prob_map[
                        unvisited_state][0]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] != self.reward_map_copy[self.visited_goal_state - self.output_states_offset]:
                        self.reward_states_prob_map[indx][0] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.reward_states_prob_map[indx][0] > 1.0:
                            self.reward_states_prob_map[indx][0] = 1.0
                        self.reward_states_prob_map[indx][1] = 1 - self.reward_sates_prob_map[indx][0]
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                unvisited_state_list = []
                for any_state in range(len(self.reward_map_copy)):
                    if self.reward_map_copy[any_state] != self.output_states_copy[self.visited_goal_state - self.output_states_offset]:
                        unvisited_state_list.append(any_state)
                for unvisited_state in unvisited_state_list:
                    self.reward_prob_map[unvisited_state][0] *= max(0, 1/self.decay_rate - 0.1 + 0.2*random_number)
                    if self.reward_prob_map[unvisited_state][0] > 1.0:
                        self.reward_prob_map[unvisited_state][0] = 1.0
                    self.reward_prob_map[unvisited_state][1] = 1.0 - self.reward_prob_map[
                        unvisited_state][0]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] != self.output_states_copy[self.visited_goal_state - self.output_states_offset]:
                        self.reward_states_prob_map[indx][0] *= max(0, 1/self.decay_rate - 0.1 + 0.2*random_number)
                        if self.reward_states_prob_map[indx][0] > 1.0:
                            self.reward_states_prob_map[indx][0] = 1.0
                        self.reward_states_prob_map[indx][1] = 1 - self.reward_sates_prob_map[indx][0]
                self.state_reward_func = self._make_state_reward_func()


    def restore_rwd(self):
        """
            restore decayed reward value of unvisited states
        """
        if self.visited_goal_state > 0 and random.random()> self.restore_drop_rate:
            if self.nine_states_mode == 1:
                random_number = random.random()
                unvisited_state_list = []
                for any_state in range(len(self.reward_map)):
                    if self.reward_map_copy[any_state] != self.reward_map_copy[self.visited_goal_state - self.output_states_offset]:
                        unvisited_state_list.append(any_state)
                for unvisited_state in unvisited_state_list:
                    self.reward_map[unvisited_state] *= max(0, 1 / np.power(self.decay_rate,
                                                                            0.5) - 0.1 + 0.2 * random_number)
                    if self.reward_map[unvisited_state] > self.reward_map_copy[unvisited_state]:
                        self.reward_map[unvisited_state] = self.reward_map_copy[unvisited_state]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] != self.reward_map_copy[self.visited_goal_state - self.output_states_offset]:
                        self.output_states[indx] *= max(0,
                                                        1 / np.power(self.decay_rate, 0.5) - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                unvisited_state_list = []
                for any_state in range(len(self.reward_map)):
                    if self.reward_map_copy[any_state] != self.output_states_copy[self.visited_goal_state - self.output_states_offset]:
                        unvisited_state_list.append(any_state)
                for unvisited_state in unvisited_state_list:
                    self.reward_map[unvisited_state] *= max(0, 1/np.power(self.decay_rate,0.5) - 0.1 + 0.2*random_number)
                    if self.reward_map[unvisited_state] > self.reward_map_copy[unvisited_state]:
                        self.reward_map[unvisited_state] = self.reward_map_copy[unvisited_state]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] != self.output_states_copy[self.visited_goal_state- self.output_states_offset]:
                        self.output_states[indx] *= max(0, 1/np.power(self.decay_rate,0.5) - 0.1 + 0.2*random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()

    def restore_rwd_specific(self,goal_state):
        """
            restore decayed reward value of visited states
        """
        if random.random() > self.restore_drop_rate:
            if self.nine_states_mode == 1:
                random_number = random.random()
                reward_map_indx = goal_state - self.output_states_offset
                self.reward_map[reward_map_indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                if self.reward_map[reward_map_indx] > self.reward_map_copy[reward_map_indx]:
                    self.reward_map[reward_map_indx] = self.reward_map_copy[reward_map_indx]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                reward_map_indx = self.reward_map_copy.index(
                    self.output_states_copy[goal_state - self.output_states_offset])
                self.reward_map[reward_map_indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                if self.reward_map[reward_map_indx] > self.reward_map_copy[reward_map_indx]:
                    self.reward_map[reward_map_indx] = self.reward_map_copy[reward_map_indx]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()

    def restore_visited_rwd(self):
        """
            restore decayed reward value of visited states
        """
        if self.visited_goal_state > 0 and random.random()> self.restore_drop_rate:
            if self.nine_states_mode == 1:
                random_number = random.random()
                reward_map_indx = self.visited_goal_state - self.output_states_offset
                self.reward_map[reward_map_indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                if self.reward_map[reward_map_indx] > self.reward_map_copy[reward_map_indx]:
                    self.reward_map[reward_map_indx] = self.reward_map_copy[reward_map_indx]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0, 1 / self.decay_rate - 0.1 + 0.2 * random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()
            else:
                random_number = random.random()
                reward_map_indx = self.reward_map_copy.index(
                    self.output_states_copy[self.visited_goal_state - self.output_states_offset])
                self.reward_map[reward_map_indx] *= max(0, 1/self.decay_rate - 0.1 + 0.2*random_number)
                if self.reward_map[reward_map_indx] > self.reward_map_copy[reward_map_indx]:
                    self.reward_map[reward_map_indx] = self.reward_map_copy[reward_map_indx]
                self.reward_map_func = self._make_reward_map_func()
                for indx in range(len(self.output_states)):
                    if self.output_states_copy[indx] == self.reward_map_copy[reward_map_indx]:
                        self.output_states[indx] *= max(0, 1/self.decay_rate - 0.1 + 0.2*random_number)
                        if self.output_states[indx] > self.output_states_copy[indx]:
                            self.output_states[indx] = self.output_states_copy[indx]
                self.state_reward_func = self._make_state_reward_func()

    def shift_high_low_uncertainty(self):
        if self.trans_prob[0] == 0.5:
            self.trans_prob = [0.9, 0.1]
        elif self.trans_prob[0] == 0.9:
            self.trans_prob = [0.5, 0.5]
        else:
            raise ValueError

    def reward_contrast(self):
        self.restore_rwd()
        self.decay_rwd()

    def reward_prob_contrast(self):
        self.restore_prob()
        self.decay_prob()

    def tree_ladder_shift(self):

        None