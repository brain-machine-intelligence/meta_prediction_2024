import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
from scipy.interpolate import pchip_interpolate
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io as sio
matplotlib.use('Agg')
MODE_LIST = ['min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe']
MODE_MAP = {
    'min-spe' : ['spe', None, 'red', 'MIN_SPE'],
    'max-spe' : ['spe', None, 'mediumseagreen', 'MAX_SPE'],
    'min-rpe' : ['rpe', None, 'royalblue', 'MIN_RPE'],
    'max-rpe' : ['rpe', None, 'plum', 'MAX_RPE'],
    'min-rpe-min-spe' : ['spe', 'rpe', 'tomato', 'MIN_RPE_MIN_SPE'],
    'max-rpe-max-spe' : ['spe', 'rpe', 'dodgerblue', 'MAX_RPE_MAX_SPE'],
    'max-rpe-min-spe' : ['spe', 'rpe', 'y', 'MAX_RPE_MIN_SPE'],
    'min-rpe-max-spe' : ['spe', 'rpe', 'mediumvioletred', 'MIN_RPE_MAX_SPE']
}
VAR_MAP_LIST = ['PMB','SPE', 'RPE','RWD','SCR']
#folderpath = '20221021'  # RPE SPE task controller for publication
#file_suffix = '_2021_20_trials_20221010_delta_control_highest'   # RPE SPE task controller for publication
folderpath = '20230627'  # SPE task controller with reward restore dropout
file_suffix = '_2021_20_trials_20230627_delta_control_restore_drop_highest'  # SPE task controller with reward restore dropout
folderpath = '20230926'
file_suffix = '_2021_20_trials_20230926_delta_control_restore_drop_highest'
folderpath = '20230927'
file_suffix = '_2021_20_trials_20230927_delta_control_restore_drop_highest'
folderpath = '20231020'
file_suffix = '_2021_20_trials_20240717_delta_control_restore_drop_highest'
folderpath = '20240717'
#file_suffix = '_2014_20_trials_20231020_delta_control_restore_drop_highest'
if file_suffix == '_2021_20_trials_20230926_delta_control_restore_drop_highest' or file_suffix == '_2021_20_trials_20230927_delta_control_restore_drop_highest'\
        or file_suffix == '_2014_20_trials_20231020_delta_control_restore_drop_highest':
    MODE_LIST = ['min-MF','max-MF','min-MB','max-MB','min-MF-min-MB', 'max-MF-max-MB', 'max-MF-min-MB', 'min-MF-max-MB']
    MODE_MAP = {
        'min-MB': ['MB', None, 'red', 'MIN_MB'],
        'max-MB': ['MB', None, 'mediumseagreen', 'MAX_MB'],
        'min-MF': ['MF', None, 'royalblue', 'MIN_MF'],
        'max-MF': ['MF', None, 'plum', 'MAX_MF'],
        'min-MF-min-MB': ['MB', 'MF', 'tomato', 'MIN_MF_MIN_MB'],
        'max-MF-max-MB': ['MB', 'MF', 'dodgerblue', 'MAX_MF_MAX_MB'],
        'max-MF-min-MB': ['MB', 'MF', 'y', 'MAX_MF_MIN_MB'],
        'min-MF-max-MB': ['MB', 'MF', 'mediumvioletred', 'MIN_MF_MAX_MB']
    }
    #VAR_MAP_LIST = ['PMB', 'SPE', 'RPE', 'RWD', 'SCR','MBR','MFR']
    VAR_MAP_LIST = ['STT','ACT']
max_sbj = 82

if file_suffix == '_2021_20_trials_20230927_delta_control_restore_drop_highest':
    folderpath = '20230927'
elif file_suffix == '_2014_20_trials_20231020_delta_control_restore_drop_highest':
    folderpath = '20231020'

file_suffix = '_2023_20_trials_20231214_delta_control_restore_drop_highest'
file_suffix = '_2021_20_trials_20231214_delta_control_restore_drop_highest'
file_suffix = '_2020_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_2020_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_2014_20_trials_20231214_delta_control_restore_drop_highest'
#file_suffix = '_20201_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20202_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20203_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20211_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20212_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20213_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20214_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_20215_20_trials_20240110ab_delta_control_restore_drop_highest'
#file_suffix = '_2021_20_trials_20240717_delta_control_restore_drop_highest'
#tasks = ['20211','20212','20213','20214','20215','20201','20202','20203']
#tasks = ['20211','20201']
tasks = ['2020']
tasks = ['20201','20202','20203']
#tasks = ['20211','20212','20213','20214','20215']

for task_type in tasks:
    
    ablation_tasks = ["20201","20202","20203","20211","20212","20213","20214","20215"]
    if task_type in ablation_tasks:
        file_suffix = '_'+task_type+'_20_trials_20240725_delta_control_restore_drop_highest' ## Ablation
    if '_20_trials_20240725_delta_control_restore_drop_highest' in file_suffix:
        folderpath = '20240725'
    elif file_suffix == '_2023_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214' #SPE task with transition probability action change ablation
    elif file_suffix == '_2021_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214' #previous SPE tasks
    elif file_suffix == '_2020_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # previous RPE tasks
    elif file_suffix == '_2014_20_trials_20231214_delta_control_restore_drop_highest':
        folderpath = '20231214'  # previous ori tasks
    elif file_suffix == '_2021_20_trials_20240717_delta_control_restore_drop_highest':
        folderpath = '20240717'
    elif any(x in file_suffix for x in ablation_tasks):
        folderpath = '20240110ab'


    MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                'max-MF-min-MB', 'min-MF-max-MB']
    MODE_MAP = {
        'min-spe' : ['spe', None, 'red', 'MIN_SPE'],
        'max-spe' : ['spe', None, 'mediumseagreen', 'MAX_SPE'],
        'min-rpe' : ['rpe', None, 'royalblue', 'MIN_RPE'],
        'max-rpe' : ['rpe', None, 'plum', 'MAX_RPE'],
        'min-rpe-min-spe' : ['spe', 'rpe', 'tomato', 'MIN_RPE_MIN_SPE'],
        'max-rpe-max-spe' : ['spe', 'rpe', 'dodgerblue', 'MAX_RPE_MAX_SPE'],
        'max-rpe-min-spe' : ['spe', 'rpe', 'y', 'MAX_RPE_MIN_SPE'],
        'min-rpe-max-spe' : ['spe', 'rpe', 'mediumvioletred', 'MIN_RPE_MAX_SPE'],
        'min-MB': ['MB', None, 'red', 'MIN_MB'],
        'max-MB': ['MB', None, 'mediumseagreen', 'MAX_MB'],
        'min-MF': ['MF', None, 'royalblue', 'MIN_MF'],
        'max-MF': ['MF', None, 'plum', 'MAX_MF'],
        'min-MF-min-MB': ['MB', 'MF', 'tomato', 'MIN_MF_MIN_MB'],
        'max-MF-max-MB': ['MB', 'MF', 'dodgerblue', 'MAX_MF_MAX_MB'],
        'max-MF-min-MB': ['MB', 'MF', 'y', 'MAX_MF_MIN_MB'],
        'min-MF-max-MB': ['MB', 'MF', 'mediumvioletred', 'MIN_MF_MAX_MB']
    }

    rpe_abl = ["20201","20202","20203"]
    spe_abl = ["20211","20212","20213","20214","20215"]
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
        NUM_ACT = len(action_list)
        pol_list = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                    'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                    'max-MF-min-MB', 'min-MF-max-MB']
        MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'min-rpe-max-spe',
                    'max-rpe-min-spe', 'min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                    'max-MF-min-MB', 'min-MF-max-MB']
        POL_TAG = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
        pol_list_tick = ['r', 'R', 's', 'S', 'rs', 'RS', 'rS', 'Rs', 'f', 'F', 'b', 'B', 'fb', 'FB', 'fB', 'Fb']
    elif any(x in file_suffix for x in rpe_abl):
        MODE_LIST = ['min-rpe', 'max-rpe']
        MODE_MAP = {
            'min-rpe': ['rpe', None, 'royalblue', 'MIN_RPE'],
            'max-rpe': ['rpe', None, 'plum', 'MAX_RPE']
        }
    elif any(x in file_suffix for x in spe_abl):
        MODE_LIST = ['min-spe', 'max-spe']
        MODE_MAP = {
            'min-spe': ['spe', None, 'red', 'MIN_SPE'],
            'max-spe': ['spe', None, 'mediumseagreen', 'MAX_SPE']
        }
    elif file_suffix == '_2021_20_trials_20240717_delta_control_restore_drop_highest':
        MODE_LIST = ['min-MF', 'max-MF', 'min-MB', 'max-MB', 'min-MF-min-MB', 'max-MF-max-MB',
                     'max-MF-min-MB', 'min-MF-max-MB']
        MODE_MAP = {
            'min-MB': ['MB', None, 'red', 'MIN_MB'],
            'max-MB': ['MB', None, 'mediumseagreen', 'MAX_MB'],
            'min-MF': ['MF', None, 'royalblue', 'MIN_MF'],
            'max-MF': ['MF', None, 'plum', 'MAX_MF'],
            'min-MF-min-MB': ['MB', 'MF', 'tomato', 'MIN_MF_MIN_MB'],
            'max-MF-max-MB': ['MB', 'MF', 'dodgerblue', 'MAX_MF_MAX_MB'],
            'max-MF-min-MB': ['MB', 'MF', 'y', 'MAX_MF_MIN_MB'],
            'min-MF-max-MB': ['MB', 'MF', 'mediumvioletred', 'MIN_MF_MAX_MB']   
        }

    #file_suffix = '_2020_delta_trials_control_highest'
    #file_suffix = '_2019_delta_trials_control_highest'
    # if file_suffix == '_2020_delta_trials_control' or file_suffix == '_2020_20_trials_delta_control_highest':
    #    MODE_LIST = ['min-rpe', 'max-rpe']

    VAR_MAP_LIST = ['PMB', 'SPE', 'RPE', 'RWD', 'SCR','MBR','MFR']
    #VAR_MAP_LIST = ['STT','ACT']
    #VAR_MAP_LIST = ['ENV']
    mode_idf = 2 #2-min 3-max
    do_anova = False
    var_idf = 0
    CONTROL_resting = 99

    FBA=np.load('FBA.npy')
    FBA_lb = np.zeros(max_sbj)
    for ii in range(max_sbj):
        #if FBA[ii]<np.sort(FBA)[27]:
        if FBA[ii]<0.6:
            FBA_lb[ii]=0
        else:
            FBA_lb[ii] = 1

    pol_sbj_map = np.zeros((max_sbj,max_sbj))
    for ii in range(max_sbj):
        for jj in range(max_sbj):
                pol_sbj_map[ii][jj]=ii

    NUM_EPISODES = 101
    TRIALS_PER_EPISODE = 20

    print(folderpath + '/' + file_suffix)

    for mode_idf in range(len(MODE_LIST)):
        for var_idf in range(len(VAR_MAP_LIST)):
            print(VAR_MAP_LIST[var_idf] +' result in the '+ MODE_LIST[mode_idf])
            VAR_MAP=np.zeros((max_sbj,max_sbj,TRIALS_PER_EPISODE*(NUM_EPISODES+CONTROL_resting)))
            if VAR_MAP_LIST[var_idf] in ['STT','ACT']:
                VAR_MAP = np.zeros((max_sbj, max_sbj, TRIALS_PER_EPISODE * (NUM_EPISODES + CONTROL_resting) * 2))
            elif VAR_MAP_LIST[var_idf] == 'ENV':
                VAR_MAP = np.zeros((max_sbj, max_sbj, TRIALS_PER_EPISODE * (NUM_EPISODES + CONTROL_resting),4))
            VAR_statics = np.zeros((max_sbj,max_sbj))
            MatchNonMatch = np.zeros((max_sbj*max_sbj))
            for policy_sbj_indx in range(max_sbj):
                #MatchNonMatch[policy_sbj_indx*max_sbj+policy_sbj_indx]=1
                VAR_MAP[policy_sbj_indx]=np.load('history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
                for affected_sbj_indx in range(max_sbj):
                    if FBA_lb[policy_sbj_indx] == FBA_lb[affected_sbj_indx] :
                        MatchNonMatch[policy_sbj_indx*max_sbj+affected_sbj_indx]=1
                    if VAR_MAP_LIST[var_idf] == 'ENV':
                        VAR_statics[policy_sbj_indx][affected_sbj_indx]=sum(VAR_MAP[policy_sbj_indx,affected_sbj_indx,:,0])/len(VAR_MAP[policy_sbj_indx,affected_sbj_indx,:,0])
                    else:
                        VAR_statics[policy_sbj_indx][affected_sbj_indx] = sum(
                            VAR_MAP[policy_sbj_indx][affected_sbj_indx]) / len(VAR_MAP[policy_sbj_indx][affected_sbj_indx])

            if do_anova == True:
                df = pd.DataFrame(data=VAR_statics, index=range(max_sbj), columns=range(max_sbj))
                df2 = pd.melt(df.reset_index(), id_vars=['index'], value_vars=range(max_sbj))
                df2['match'] = MatchNonMatch
                formula = 'value ~ C(match)'
                lm = ols(formula, df2).fit()
                print(anova_lm(lm))


            ori_mean = np.zeros(TRIALS_PER_EPISODE)
            ori_sem = np.zeros(TRIALS_PER_EPISODE)
            shu_mean = np.zeros(TRIALS_PER_EPISODE)
            shu_sem = np.zeros(TRIALS_PER_EPISODE)
            RWD_MAP = np.zeros((max_sbj,max_sbj,TRIALS_PER_EPISODE))
            for policy_sbj_indx in range(max_sbj):
                temp = np.load('history_results/' + folderpath + '/SUB{0:03d}_SHUFFLE_'.format(policy_sbj_indx) + MODE_LIST[mode_idf] + file_suffix + '_' + VAR_MAP_LIST[var_idf] + '.npy')
                for affected_sbj_indx in range(max_sbj):
                    for episode in range(TRIALS_PER_EPISODE):
                        if VAR_MAP_LIST[var_idf] == 'ENV':
                            RWD_MAP[policy_sbj_indx][affected_sbj_indx][episode]=np.mean(temp[affected_sbj_indx,TRIALS_PER_EPISODE*(episode+CONTROL_resting):TRIALS_PER_EPISODE*(episode+CONTROL_resting+1),0])
                        else:
                            RWD_MAP[policy_sbj_indx][affected_sbj_indx][episode]=np.mean(temp[affected_sbj_indx][TRIALS_PER_EPISODE*(episode+CONTROL_resting):TRIALS_PER_EPISODE*(episode+CONTROL_resting+1)])

            tmp_ori_full = []
            for trials in range(TRIALS_PER_EPISODE):
                tmp_ori = np.zeros(max_sbj*NUM_EPISODES)
                tmp_shu = np.zeros((max_sbj-1)*max_sbj*NUM_EPISODES)
                tmp_ori_indx = 0
                tmp_shu_indx = 0
                for policy_sbj_indx in range(max_sbj):
                    for affected_sbj_indx in range(max_sbj):
                        opt_eps = np.argmax(RWD_MAP[policy_sbj_indx][affected_sbj_indx])
                        for eps in range(NUM_EPISODES):
                            if policy_sbj_indx==affected_sbj_indx:
                                if VAR_MAP_LIST[var_idf] == 'ENV':
                                    tmp_ori[tmp_ori_indx]=VAR_MAP[policy_sbj_indx,affected_sbj_indx,trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE,0]
                                else:
                                    tmp_ori[tmp_ori_indx]=VAR_MAP[policy_sbj_indx][affected_sbj_indx][trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE]
                                tmp_ori_indx += 1
                            else:
                                if VAR_MAP_LIST[var_idf] == 'ENV':
                                    tmp_shu[tmp_shu_indx]=VAR_MAP[policy_sbj_indx][affected_sbj_indx][trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE,0]
                                else:
                                    tmp_shu[tmp_shu_indx]=VAR_MAP[policy_sbj_indx][affected_sbj_indx][trials+TRIALS_PER_EPISODE*eps+CONTROL_resting*TRIALS_PER_EPISODE]
                                tmp_shu_indx += 1
                ori_mean[trials] = np.mean(tmp_ori)
                ori_sem[trials] = np.std(tmp_ori) / np.sqrt(len(tmp_ori))
                shu_mean[trials] = np.mean(tmp_shu)
                shu_sem[trials] = np.std(tmp_shu) / np.sqrt(len(tmp_shu))
                tmp_ori_full.append(tmp_ori)
           # print(ori_mean)
            print(str(np.mean(ori_mean)) + " " + str(np.std(tmp_ori_full)/np.sqrt(len(tmp_ori_full)-1)))
            TICK_NAME_NUM = np.linspace(1, TRIALS_PER_EPISODE + 1, TRIALS_PER_EPISODE+1)[:-1]
            TICK_NAME_STR = [str(x) for x in TICK_NAME_NUM]
            smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
            smooth_y = pchip_interpolate(TICK_NAME_NUM, ori_mean, smooth_x)
            ax = plt.gca()
            ax.plot(smooth_x, smooth_y, label='original', color='red')
            ax.set_xlabel('Game step in one policy')
            ax.set_ylabel('original ' + VAR_MAP_LIST[var_idf] + 'control')
            smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, ori_sem, smooth_x))
            ax.fill_between(smooth_x, smooth_y - 1.96 * smooth_sem, smooth_y + 1.96 * smooth_sem, alpha=0.2,
                            color='red')
            plt.legend(bbox_to_anchor=(1.02, 1))


            smooth_x = np.linspace(0, TICK_NAME_NUM[-1], 200)
            smooth_y = pchip_interpolate(TICK_NAME_NUM, shu_mean, smooth_x)
            ax.plot(smooth_x, smooth_y, label='shuffled policy', color='black')
            ax.set_xlabel('Game step in one policy')
            ax.set_ylabel('shuffled ' + VAR_MAP_LIST[var_idf] + ' control')
            smooth_sem = pd.Series(data=pchip_interpolate(TICK_NAME_NUM, shu_sem, smooth_x))
            ax.fill_between(smooth_x, smooth_y - 1.96 * smooth_sem, smooth_y + 1.96 * smooth_sem, alpha=0.2, color='black')
            plt.legend(bbox_to_anchor=(1.02, 1))
            #if var_idf == 2: plt.ylim(6,10)
            plt.savefig('history_results/' + folderpath + '/' + 'Shuffled  policy '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
            plt.clf()

            plt.scatter(pol_sbj_map.flatten(),np.median(RWD_MAP,axis=2).flatten())
            plt.savefig('history_results/' + folderpath + '/' + 'Shuffled policy-specific '+ VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] + file_suffix +'.png')
            plt.clf()


            sio.savemat('history_results/' + folderpath + '/' + VAR_MAP_LIST[var_idf] +' result in the '+ MODE_MAP[MODE_LIST[mode_idf]][3] +"data" + file_suffix + ".mat", {'data': VAR_MAP})



