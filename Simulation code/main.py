import getopt
import sys
import csv
import os
import simulation as sim
import analysis
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import torch

from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
# from training_layer2 import Net
from torch.autograd import Variable

usage_str = """
Model-free, model-based learning simulation

Usage:

Simulation control parameters:

    -d load parameters from csv file, default is regdata.csv

    -n [number of parameters entries to simulate]

    --episodes [num episodes]

    --trials [num trials per episodes]

    --set-param-file [parameter file]             Specify the parameter csv file to load

    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment

    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe/min-mf-rel/max-mf-rel/min-mb-rel/max-mb-rel> 
                                                  Choose control agent mode

    --legacy-mode                                 Use legacy MDP environment, which treats one type of terminal reward as one
                                                  state. Since C++ ext is not implemented for this type of environment, legacy
                                                  pure Python implemenation for FORWARD will be use.

    --disable-control                             Disable control agents

    --all-mode                                    Execute all control mode

    --enable-static-control                       Use static control instead of DDQN control

    --disable-detail-plot                         Disable plot for each simulation

    --disable-c-ext                               Disable using C extension

    --less-control-input                          Less environment input for control agent

    --save-ctrl-rl                                Save control RL agent object for further use

    --no-reset                                    Not reset environment and agent when start every episode. Default is reset

    --save-log-Q-value                            Save action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator) in gData as a pickle file.
                                                  Default is not saving. 
                                                  Only possible in '--all-mode -d'

Analysis control parameters:

    --re-analysis [analysis object pickle file]   Re-run analysis functions

    --PCA-plot                                    Generate plot against PCA results. Not set by default because
                                                  previous PCA run shows MB preference gives 99% variance, so comparing 
                                                  against MB preference is good enough, instead of some principal component

    --learning-curve-plot                         Plot learning curves

    --use-confidence-interval                     When plot with error bar, use confidence interval instead of IQR

    --separate-learning-curve                     Separate learning curve plot

    --disable-auto-max                            Use fix max value on y axis when plotting learning curve in one episode     

    --to-excel [subject id]                       Generate a excel file for specific subject with detail sequence of data

    --disable-action-compare                      Use actions as feature space

    --enable-score-compare                        Use score as feature space

    --human-data-compare                          Enable comparison against the full columns of human data

    --use-selected-subjects                       Use selected subjects, defualt is min 25 50 75 max five subjects

    --head-tail-subjects                          Use head and tail subjects to emphasize the difference

    --cross-mode-plot                             Plots that compare data between modes

    --enhance-compare <boost/inhibit/cor/sep>     Only plot two modes depending on the scenario to compare

    --cross-compare [mode]                        Extract best action sequence from subject A in a given mode. Apply to subject B.
                                                  Plot against subject B's original data

    --opposite-cross-compare [mode]               Extract worst action sequence(opposite mode optimal sequence) from specific subject in a given mode. 
                                                  Apply to that subject. Plt against the subject's original data

    --sub-A [subject A]                           Subject A for cross compare // useless for current version code

    --sub-B [subject B]                           Subject B for cross compare // useless for current version code

    --to-excel-log-Q-value                        Generate a excel file for action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator) in pickle file
                                                  To use this option, '--save-log-Q-value' option should be used before 
    
    --to-excel-random-mode-sequence               Generate a excel file of random mode sequence after executing mixed-random mode


    --to-excel-optimal-sequence                   Generate a excel file of optimal sequence.

"""

def usage():
    print(usage_str)

LOAD_PARAM_FILE   = False
NUM_PARAMETER_SET = 82
ALL_MODE          = False
ANALYSIS_OBJ      = None
TO_EXCEL          = None
SCENARIO          = None
CROSS_MODE_PLOT   = False
CROSS_COMPARE     = False
OPPOSITE_CROSS_COMPARE = False
OPPOSITE_NN_CROSS_COMPARE = False
FAIR_OPPOSITE_NN_CROSS_COMPARE = False
CROSS_COMPARE_MOD = 'min-spe'
SUBJECT_A         = 10 # Low MB->MF trans rate
SUBJECT_B         = 17 # High MB->MF trans rate
#PARAMETER_FILE    = '82nd_subj.csv'
PARAMETER_FILE    = 'regdata.csv'
TO_EXCEL_LOG_Q_VALUE = False
TO_EXCEL_RANDOM_MODE_SEQUENCE = False
TO_EXCEL_OPTIMAL_SEQUENCE = False
FILE_SUFFIX = ''
SCENARIO_MODE_MAP = {
    'boost'   : ['min-spe', 'min-rpe'],
    'inhibit' : ['min-spe', 'max-spe'],
    'cor'     : ['min-rpe-min-spe', 'max-rpe-max-spe'],
    'sep'     : ['min-rpe-max-spe', 'max-rpe-min-spe']
}
ORIGINAL_MODE_MAP = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
OPPOSITE_MODE_MAP = ['max-spe', 'min-spe', 'max-rpe', 'min-rpe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'min-rpe-max-spe', 'max-rpe-min-spe']

MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe', 'min-rpe-max-spe','min-rpe-PMB','max-rpe-PMB','min-rpe-max-spe-PMB', 'random']
MODE_LIST_INPUT = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [1, -1], [-1, 1],[-1,0],[1,0],[0,0]]


def reanalysis(analysis_object):
    with open(analysis_object, 'rb') as pkl_file:
        gData = pickle.load(pkl_file)
    with open(PARAMETER_FILE) as f:
        csv_parser = csv.reader(f)
        param_list = []
        for row in csv_parser:
            param_list.append(tuple(map(float, row[:-1])))
    if CROSS_MODE_PLOT:
        if SCENARIO is not None:
            gData.cross_mode_summary(SCENARIO_MODE_MAP[SCENARIO])
        else:
            gData.cross_mode_summary()
    elif CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = [mode for mode, _ in MODE_MAP.items()]
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        for compare_mode in mode_iter_lst:
            SUBJECT_A = 81 # default change this to policy number you want to apply
            for SUBJECT_B in range(NUM_PARAMETER_SET):
                #SUBJECT_A, SUBJECT_B = choice(82, 2, replace=False)
                # set up simulation with static control sequence from subject A
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[compare_mode] = gData.get_optimal_control_sequence(compare_mode, SUBJECT_A)
                sim.CONTROL_MODE = compare_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[SUBJECT_B]), PARAMETER_SET=str(SUBJECT_B), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df
                gData.cross_mode_summary([analysis.MODE_IDENTIFIER, compare_mode], [0, SUBJECT_B], [SUBJECT_A, SUBJECT_B])
                gData.plot_transfer_compare_learning_curve(compare_mode, SUBJECT_B, SUBJECT_A)
    elif OPPOSITE_CROSS_COMPARE:
        #compare current policy with opposite policy
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[ORIGINAL_MODE_MAP.index(current_mode)] # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df
                gData.cross_mode_summary2([analysis.MODE_IDENTIFIER2, current_mode], [0, subject], [subject, subject])
                gData.plot_opposite_compare_learning_curve(current_mode, subject)
    elif OPPOSITE_NN_CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        #load neural network
        net = torch.load('ControlRL/Net_140output100_best')
        net.eval() 
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[ORIGINAL_MODE_MAP.index(current_mode)] # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                #For opposite sequence
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df

                #for optimal sequence from Neural Net
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                # we should make neural net function to draw figure
                idx = MODE_LIST.index(current_mode)
                net_input = Variable(torch.FloatTensor(list(param_list[subject])[:-1] + MODE_LIST_INPUT[idx] ))
                output = net(net_input[None, ...])
                output = output[0]
                decoded_output = []
                for i in range(0,len(output),4):
                    #print(torch.argmax(test_output[i:i+4]))
                    tensor = torch.argmax(output[i:i+4])
                    decoded_output.append( tensor.item() )
                print(net_input, decoded_output)
                sim.static_action_map[current_mode] = decoded_output#[0]*sim.TRIALS_PER_EPISODE #gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df                


                gData.cross_mode_summary3([analysis.MODE_IDENTIFIER, analysis.MODE_IDENTIFIER2, current_mode], [0, 0, subject], [subject, subject, subject])
                gData.plot_opposite_nn_compare_learning_curve(current_mode, subject) 
    elif FAIR_OPPOSITE_NN_CROSS_COMPARE:
        # human random exploration of 200 episode is done for fair comparison.
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        #load neural network
        net = torch.load('ControlRL/Net_140output100_best')
        net.eval() 
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[ORIGINAL_MODE_MAP.index(current_mode)] # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                #For opposite sequence
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 200
                sim.static_action_map[compare_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = compare_mode
                sim.ENABLE_PLOT = False
                sim.RESET = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df

                #For optimal sequence from Neural Net
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 200
                # we should make neural net function to draw figure
                idx = MODE_LIST.index(current_mode)
                net_input = Variable(torch.FloatTensor(list(param_list[subject])[:-1] + MODE_LIST_INPUT[idx] ))
                output = net(net_input[None, ...])
                output = output[0]
                decoded_output = []
                for i in range(0,len(output),4):
                    #print(torch.argmax(test_output[i:i+4]))
                    tensor = torch.argmax(output[i:i+4])
                    decoded_output.append( tensor.item() )
                print(net_input, decoded_output)
                sim.static_action_map[current_mode] = decoded_output#[0]*sim.TRIALS_PER_EPISODE #gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                sim.RESET = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df                

                #For optimal sequence from simulation
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 200
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(current_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                sim.RESET = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject), return_res=True)
                gData.data[analysis.MODE_IDENTIFIER3] = [None]
                gData.detail[analysis.MODE_IDENTIFIER3] = [None]
                gData.data[analysis.MODE_IDENTIFIER3][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER3][0] = res_detail_df

                gData.cross_mode_summary4([analysis.MODE_IDENTIFIER, analysis.MODE_IDENTIFIER2, analysis.MODE_IDENTIFIER3], [0, 0, 0], [subject, subject, subject], current_mode)
                gData.plot_opposite_nn_compare_learning_curve(current_mode, subject)            
    else:
        for mode, _ in tqdm(MODE_MAP.items()):
            try:
                gData.set_current_mode(mode)
                gData.generate_summary(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')
        mode = 'mixed-random'
        try:
            gData.set_current_mode(mode)
            gData.generate_summary(mode)
        except KeyError:
            print('mode: ' + mode + ' data not found. Skip')
    if TO_EXCEL is not None:
        print("Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column = 'action')
    if TO_EXCEL is not None:
        print("Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column = 'ctrl_reward')
    if TO_EXCEL is not None:
        print("Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column = 'score')
    if TO_EXCEL_LOG_Q_VALUE:
        print("Making excel file of human action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator)")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.log_Q_value_to_excel(i)
    if TO_EXCEL_RANDOM_MODE_SEQUENCE:
        gData.random_mode_sequence_to_excel()

    if TO_EXCEL_OPTIMAL_SEQUENCE:
        for mode, _ in tqdm(MODE_MAP.items()):
            try:
                gData.optimal_sequence_to_excel(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')
            

if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt  = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=", "all-mode", "enable-static-control",
                 "disable-c-ext", "disable-detail-plot", "less-control-input", "re-analysis=", "PCA-plot", "learning-curve-plot", "use-confidence-interval",
                 "to-excel=", "disable-action-compare", "enable-score-compare", "use-selected-subjects", "save-ctrl-rl", "head-tail-subjects", 
                 "human-data-compare", "disable-auto-max", "legacy-mode", "separate-learning-curve", "cross-mode-plot", "cross-compare=", "sub-A=", "sub-B=",
                 "enhance-compare=" , "no-reset", "save-log-Q-value", "to-excel-log-Q-value", "mixed-random-mode", "to-excel-random-mode-sequence", 
                 "to-excel-optimal-sequence", "opposite-cross-compare=", "opposite-nn-cross-compare=", "fair-opposite-nn-cross-compare=", "file-suffix=", "task-type=",
                 "PMB_CONTROL=", "Reproduce_BHV=", "Session_block=", "mode202010", 'delta-control=', 'restore-drop-rate=' ,'action-prob']
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o == "--disable-control":
            sim.CTRL_AGENTS_ENABLED = False
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o == "--mdp-stages":
            sim.MDP_STAGES = int(a)
        elif o == "--ctrl-mode":
            sim.CONTROL_MODE = a
            sim.CONTROL_MODES_LIST = sim.CONTROL_MODE
        elif o == "-d":
            LOAD_PARAM_FILE = True
        elif o == "--set-param-file":
            PARAMETER_FILE = a
        elif o == "--episodes":
            sim.TOTAL_EPISODES = int(a)
        elif o == "--trials":
            sim.TRIALS_PER_EPISODE = int(a)
        elif o == "--all-mode":
            ALL_MODE = True
        elif o == "-n":
            NUM_PARAMETER_SET = int(a)
            sim.NUM_PARAMETER_SET = int(a)
        elif o == "--enable-static-control":
            sim.STATIC_CONTROL_AGENT = True
        elif o == "--disable-c-ext":
            sim.DISABLE_C_EXTENSION = True
        elif o == "--legacy-mode":
            sim.DISABLE_C_EXTENSION = True
            sim.LEGACY_MODE = True
        elif o == "--disable-detail-plot":
            sim.ENABLE_PLOT = False
        elif o == "--less-control-input":
            sim.MORE_CONTROL_INPUT = False
        elif o == "--save-ctrl-rl":
            sim.SAVE_CTRL_RL = True
        elif o == "--PCA-plot":
            analysis.PCA_plot = True
        elif o == "--learning-curve-plot":
            analysis.PLOT_LEARNING_CURVE = True
        elif o == "--separate-learning-curve":
            analysis.MERGE_LEARNING_CURVE = False
        elif o == "--use-confidence-interval":
            analysis.CONFIDENCE_INTERVAL = True
        elif o == "--disable-auto-max":
            analysis.LEARNING_CURVE_AUTO_MAX = False
        elif o == "--disable-action-compare":
            analysis.ACTION_COMPARE = False
        elif o == "--enable-score-compare":
            analysis.SOCRE_COMPARE = True
        elif o == "--human-data-compare":
            analysis.HUMAN_DATA_COMPARE = True
        elif o == "--use-selected-subjects":
            analysis.USE_SELECTED_SUBJECTS = True
        elif o == "--head-tail-subjects":
            analysis.HEAD_AND_TAIL_SUBJECTS = True
        elif o == "--to-excel":
            TO_EXCEL = int(a)
        elif o == "--re-analysis":
            ANALYSIS_OBJ = a
        elif o == "--cross-mode-plot":
            CROSS_MODE_PLOT = True
        elif o == "--enhance-compare":
            SCENARIO = a
        elif o == "--cross-compare":
            CROSS_COMPARE = True
            CROSS_COMPARE_MOD = a
        elif o == "--opposite-cross-compare":
            OPPOSITE_CROSS_COMPARE = True
            CROSS_COMPARE_MOD = a
        elif o == "--opposite-nn-cross-compare":
            OPPOSITE_NN_CROSS_COMPARE = True
            CROSS_COMPARE_MOD = a
        elif o == "--fair-opposite-nn-cross-compare":
            FAIR_OPPOSITE_NN_CROSS_COMPARE = True
            CROSS_COMPARE_MOD = a
        elif o == "--sub-A":
            SUBJECT_A = int(a)
        elif o == "--sub-B":
            SUBJECT_B = int(a)
        elif o == "--no-reset":
            sim.RESET=False
        elif o == "--save-log-Q-value":
            sim.SAVE_LOG_Q_VALUE = True
        elif o == "--to-excel-log-Q-value":
            TO_EXCEL_LOG_Q_VALUE = True
        elif o =="--mixed-random-mode":
            sim.MIXED_RANDOM_MODE = True
        elif o =="--to-excel-random-mode-sequence":
            TO_EXCEL_RANDOM_MODE_SEQUENCE = True
        elif o =="--to-excel-optimal-sequence":
            TO_EXCEL_OPTIMAL_SEQUENCE = True
        elif o == "--file-suffix":
            FILE_SUFFIX = a
            sim.file_suffix = a
            print(FILE_SUFFIX)
        elif o == "--task-type":
            sim.TASK_TYPE = int(a)
        elif o == "--MF_ONLY":
            sim.MF_ONLY = bool(a)
        elif o == "--MB_ONLY":
            sim.MB_ONLY = bool(a)
        elif o == "--PMB_CONTROL":
            sim.PMB_CONTROL = bool(int(a))
        elif o == "--Reproduce_BHV":
            sim.Reproduce_BHV = bool(int(a))
            if sim.Reproduce_BHV:
                sim.saved_policy_path = 'optimal_policy'+FILE_SUFFIX+'.npy'
                FILE_SUFFIX = FILE_SUFFIX+'_repro'
        elif o == "--Session_block":
            sim.Session_block = bool(int(a))
            sim.CONTROL_MODES_LIST = sim.CONTROL_MODE
            sim.TRIALS_PER_EPISODE *= 4
            MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'random']
        elif o == "--mode202010": #ONLY MB + set learning rate as n/100, which ensure [0,1] distribution
            sim.mode202010 = True
            sim.MB_ONLY = True
            FILE_SUFFIX = FILE_SUFFIX + '_mode202010'
        elif o == "--delta-control": #RPE control task, can change hyperparameter delta for task controller
            sim.DECAY_RATE = float(a)
            FILE_SUFFIX = FILE_SUFFIX + '_delta_control'
        elif o == "--restore-drop-rate": #In restoring reward actions in controller, sometimes restoring fails
            sim.RESTORE_DROP_RATE = float(a)
            FILE_SUFFIX = FILE_SUFFIX + '_restore_drop'
        elif o == "--action-prob": #save action probability list hitory
            sim.SAVE_ACTION_PROB = True
            FILE_SUFFIX = FILE_SUFFIX + '_SAVE_APL'
        else:
            assert False, "unhandled option"

    if ANALYSIS_OBJ is not None:
        if os.path.isdir(ANALYSIS_OBJ):
            analysis_object_list = filter(lambda f: f.endswith('.pkl'), os.listdir(ANALYSIS_OBJ))
            parent_res_dir = analysis.RESULTS_FOLDER
            for index, obj in enumerate(analysis_object_list):
                analysis.RESULTS_FOLDER = parent_res_dir + '/subrun_' + str(index) + '/'
                reanalysis(os.path.join(ANALYSIS_OBJ, obj))
        else:
            reanalysis(ANALYSIS_OBJ)
        exit(0)

    gData.trial_separation = sim.TRIALS_PER_EPISODE
    if LOAD_PARAM_FILE:
        print(PARAMETER_FILE)
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            param_list = []
            for row in csv_parser:
                param_list.append(tuple(map(float, row[:-1])))
        if ALL_MODE:
            for mode, _ in MODE_MAP.items():
                gData.new_mode(sim.CONTROL_MODE)
                sim.CONTROL_MODE = mode
                print('Running mode: ' + mode)
                for index in range(NUM_PARAMETER_SET):
                    if sim.mode202010 == False:
                        print('Parameter set: ' + str(param_list[index]))
                    if sim.mode202010:
                        sim.simulation(0.514434,0.077856,0.475971,6.358767,index/100, PARAMETER_SET=str(index)) # Other parameters from the indx 0 in regdata.csv
                    else:
                        sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
#                gData.generate_summary(sim.CONTROL_MODE)
                gData.save_mode(sim.CONTROL_MODE)
                if sim.SAVE_LOG_Q_VALUE:
                    gData.save_log_Q_value(sim.CONTROL_MODE)
        elif sim.MIXED_RANDOM_MODE:
            gData.new_mode(sim.CONTROL_MODE)
            print('Running mode: mixed-random')
            for index in range(NUM_PARAMETER_SET):
                print('Parameter set: ' + str(param_list[index]))
                sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
#            gData.generate_summary('mixed-random')
            gData.save_mode('mixed-random')
            if sim.SAVE_LOG_Q_VALUE:
                gData.save_log_Q_value('mixed-random')          
        else:
            gData.new_mode(sim.CONTROL_MODE)
            print('Running mode: ' + sim.CONTROL_MODE)
            print(sim.TRIALS_PER_EPISODE)
#            for index in range(NUM_PARAMETER_SET):
            index=NUM_PARAMETER_SET
            if sim.mode202010 == False:
                print('Parameter set: ' + str(param_list[index]))
            if sim.mode202010:
                sim.simulation(0.514434, 0.077856, 0.475971, 6.358767, index / 100,
                               PARAMETER_SET=str(index))  # Other parameters from the indx 0 in regdata.csv
            else:
                sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
#            gData.generate_summary(sim.CONTROL_MODE)
            gData.save_mode(sim.CONTROL_MODE)
            if sim.SAVE_LOG_Q_VALUE:
                gData.save_log_Q_value(sim.CONTROL_MODE)
    elif ALL_MODE:
        for mode, _ in MODE_MAP.items():
            sim.CONTROL_MODE = mode
            gData.new_mode(sim.CONTROL_MODE)
            sim.simulation()
            gData.save_mode(sim.CONTROL_MODE)
    else:
        gData.new_mode(sim.CONTROL_MODE)
        sim.simulation()
        gData.save_mode(sim.CONTROL_MODE)
    
    # Save the whole analysis object for future reference
    pkl_file_name = 'Analysis-Object'
    if not ALL_MODE:
        pkl_file_name += '-'
        pkl_file_name += str(sim.CONTROL_MODE)
    pkl_file_name += '-'
    pkl_file_name += '{0:02d}'.format(NUM_PARAMETER_SET)
    print(pkl_file_name)
    with open(gData.file_name(pkl_file_name) + FILE_SUFFIX + '.pkl', 'wb') as f:
        pickle.dump(gData, f, pickle.HIGHEST_PROTOCOL)
    with open(gData.file_name(pkl_file_name) + FILE_SUFFIX+ '_NN.pkl', 'wb') as f:
        pickle.dump(gData.NN, f)
