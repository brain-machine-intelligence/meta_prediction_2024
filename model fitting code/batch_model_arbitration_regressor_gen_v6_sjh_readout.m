function [out]=batch_model_arbitration_regressor_gen_v6_sjh_readout(list_sbj_included,is_on_cluster,task_type,parameter_indx,only_40,data_type)

% this file on neuroecon is used only for optimizing the model, NOT
% generating the regressors. so the corresponding regressor files will not be saved.

% clear all
% close all
if nargin < 5 
    only_40 = false;
end


if is_on_cluster
    header='/home/sjh';
else
    header='\\143.248.30.101\sjh';
end
addpath([header, '/spm12']);

% plug-in
warning('off')
%list_sbj_included=job_opt.list_sbj_included;
%

path1=header;

switch data_type
    case '2019'
        path0=[header, '/Human_Guidance/Behavior_Simul']; % Use for 2014
        if task_type == 2019
            path0=[path0 '/task_2019'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'sbj001_SYH','sbj002_OJH','sbj003_HJH','sbj004_KMS','sbj005_LSB','sbj006_JJM','sbj007_KMJ','sbj008_KHY','sbj009_WJI','sbj010_HSJ','sbj011_PSW','sbj012_KKY','sbj013_KYL','sbj014_HSL','sbj015_MHJ','sbj016_YSH','sbj017_LHK', 'sbj018_JHH','sbj019_KHJ'...
       ,'sbj020_IYS','sbj021_JSM','sbj022_KHK','sbj023_KJS','sbj024_JHC','sbj025_JDH','sbj026_YSK','sbj027_YIJ'};
        mode.task_2022 = 0;
    case 'dep'
        path0=[header, '/Human_Guidance/Behavior_Simul']; % Use for 2014
        if task_type == 2019
            path0=[path0 '/task_2019'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'sbj1','sbj3','sbj11','sbj12','sbj18','sbj19','sbj20','sbj24','sbj25','sbj27','sbj32','sbj33','sbj34','sbj35',...                       % Use for Dep
      'sbj37','sbj38','sbj40','sbj41','sbj42','sbj44','sbj46','sbj48_kdy','sbj50_kdh','sbj51_akr','sbj52_jkw','sbj53_nyk','sbj54_jyh','sbj55_ker'}; % Use for Dep
    
    case '2014'
        path0=[header, '/fmri_arbitration/modelRLsource']; % Use for 2014
        if task_type == 2019
            path0=[path0 '/task_2019'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'Oliver', 'Hao', 'Breanna', 'Derek', 'Timothy', 'Teagan', 'Jeffrey', 'Seung', 'Carole', 'Tony', 'Surendra', 'Lark',...                                % Use for 2014
        'Joaquin', 'DavidB', 'Christopher', 'Gjergji', 'Charles', 'Erin', 'Connor', 'Domenick', 'Thao', 'Arin', 'Pauline', 'Tho'}; % Use for 2014
        mode.task_2022 = 0;
    case '2020'
        path0=[header, '/RPE_behav/Behavior_Simul']; % Use for 2014
        if task_type == 2020
            path0=[path0 '/task_2020'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'202108021500','202108051045','202108051230','202108090900','202108091415','202108091600','202108120900','202108121415'}; % Use for 2014
        mode.task_2022 = 0;
    case '2021'
        path0=[header, '/2021winter/Behavior_Simul']; % Use for 2014
        if task_type == 2020
            path0=[path0 '/task_2020'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'KSM','CJS','KCY','KMS','SHA','KAY','SHC','KDE','SYB','CYJ','KYY','YCS','PYJ','KCY2',...
            'LZY','OHJ','PYJ2','MYL','LGH','LHN','LSL','LHM','SJY','YSY','CHK','MCH','KDH'};   
        mode.task_2022 = 1;
    case '2022'
        path0=[header, '/fmri2022/Behavior_Simul']; % Use for 2014
        if task_type == 2020
            path0=[path0 '/task_2020'];
        end
        addpath(path1);
        addpath(path0);
        save_path_result=[path0 '/result_simul/']; 
        save_for_SPM=[path0, '/regressors_contrasts/'];
        save_path_neuroecon=[path0 '/regressors_contrasts/'];
        LIST_SBJ={'202210171','202210172','202210173','202210174','202210175','202210241','202210242','202210243','202210244','202210245','202210246',...
            '202210271','202210272','202210273','202210311','202210312','202210313','202210314','202210315','202211241','202211242','202211243','202211244','202211245','202211246'};  
        mode.task_2022 = 1;
end

if  ~exist(save_path_result, 'dir')
    mkdir(save_path_result);
end
if  ~exist(save_for_SPM, 'dir')
    mkdir(save_for_SPM);
end
% 1. Behavioral data
% LIST_SBJ={'david', 'DeDe', 'rosemary', 'Boyu', 'melissa', 'Rehevolew', 'joel', 'clarke', 'angela', 'william', 'josephine'}; % (good in pre which is mostly habitual - rosemary, melissa)
% mode.map_type=?;

% 2. behavioral + fmri data (for map config, see SIMUL_arbitraion_fmri2.m)
% [note] 'Oliver' uses an old map. the rest of them use a new map.
% LIST_SBJ={'CYJ'};
%LIST_SBJ_pilot={'CYJ','pilotP','pilotM'};
LIST_sbj_map_type=[1*ones(1,12) 2*ones(1,50)]; %1:'sangwan2012b', 2:'sangwan2012c'

% regressor list
% [CAUTION] DO NOT change the order!!!
% [NOTE] if "TYPE_REGRESSOR" changed, change "param_regressor_type_cue_abs_pos_in_design_mat" accordingly!!!
LIST_REGRESSOR={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'meanM1', 'meanM2', 'invFanoM1', 'invFanoM2', 'weigtM1', 'weigtM2', 'Qfwd', 'Qsarsa', 'Qarb', 'dQbwdEnergy', 'dQbwdMean', 'duncertaintyM1', 'dinvFanoM1','TR_alpha', 'TR_beta', 'dinvFano12'...
    , 'ABSdinvFano12', 'MAXinvFano12', 'CONFLICTinvFano12',' RPE_ABS', 'invF1_corrected', 'invF2_corrected', 'invF1_normalized', 'dinvFanoCorrected12'};
TYPE_REGRESSOR=[1 1, 1.5 1.5, 1.5 1.5, 1.5 1.5, 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 ...
    1.5 1.5 1.5 1.5 1.5]; % 1: parametric modulation (0-duration), 1.5:parmetric modulation (non-zero duration), 1.7:parametric modulation (with decision onset)  2: extra continuous parametric modulation (TR-fixed) - this will be used by "dummy" regressor.
row_mat=[7 7 8 8 8 8 8 8 7 7 7 7 7 7 7 8 8 7 7 8 8 8 8 ...
    7 8 8 8 8]; % from which row in the SBJ{}.regressor matrix the signal needs to be extracted. e.g., uncertainty of 0 prediction error




%% OPTION - subject
% [note] DO NOT USE sbj#[20] - he pressed wrong buttons in session1,2, so need to shrink all the SBJ matrix size by deleting the session#1,2
% list_sbj_included= job_opt.list_sbj_included;%[2:1:19 21:1:24];

%% OPTION - model optimization
option_optimizing_model=0;%job_opt.option_optimizing_model; % 0: optimizing the model for each sbj, 1: for all sbj, 2: do not optimize; load saved model
update_SBJ_structure=1; % 0: no update/just read and use, 1: update the changes to the saved SBJ file
% mode.opt_ArbModel=0;%job_opt.opt_ArbModel; % 0: full arbitrator, 1: invF-based, 2: mean-based, 3: uncertainty-based arbitrator
% mode.USE_FWDSARSA_ONLY=0;% job_opt.USE_FWDSARSA_ONLY; % 0: arbitration, 1: use fwd only, 2: use sarsa only
if parameter_indx<21
    mode.USE_FWDSARSA_ONLY=1; %fwd only
    mode.opt_ArbModel=0; % No effect
    mode.opt_m2=1;  %No effect
    mode.num_params=6;
    mode.param_seed_indx=rem(parameter_indx,20);
    if mode.param_seed_indx==0
        mode.param_seed_indx=20;
    end
    param_init=[0.5,0.2,3,3,0.1,0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
    param_BoundL= [0.1, 0.1, 1,1, 0.1 0.1];%[0.3, 0.05, 1,1 , 0.15, 0.05];
%     param_BoundL(5) = 0.5;
    param_BoundU= [0.9, 0.9, 8,10, 0.9, 0.9];%[0.75, 0.5, 8,10, 0.25, 0.15];
    mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
    if mode.param_seed_indx<17
        param_init(5) = param_BoundL(5) + (param_BoundU(5)-param_BoundL(5))*(rem(mode.param_seed_indx,4)/4+1/8);
        param_init(6) = param_BoundL(6) + (param_BoundU(6)-param_BoundL(6))*(floor(mode.param_seed_indx/4)/4+1/8);
    else        
        param_init(5)=param_BoundL(5)+(param_BoundU(5)-param_BoundL(5))*rand();
        param_init(6)=param_BoundL(6)+(param_BoundU(6)-param_BoundL(6))*rand();
    end
elseif parameter_indx<41
    mode.USE_FWDSARSA_ONLY=2;   %sarsa_only
    mode.opt_ArbModel=0; % No effect
    mode.opt_m2=1;  %No effect
    mode.num_params=6;
    mode.param_seed_indx=rem(parameter_indx,20);
    if mode.param_seed_indx==0
        mode.param_seed_indx=20;
    end
    param_init=[0.5,0.2,3,3,0.1,0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
    param_BoundL= [0.1, 0.1, 1,1, 0.1 0.1];%[0.3, 0.05, 1,1 , 0.15, 0.05];
%     param_BoundL(5) = 0.5;
    param_BoundU= [0.9, 0.9, 8,10, 0.9, 0.9];%[0.75, 0.5, 8,10, 0.25, 0.15];
    mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
    if mode.param_seed_indx<17
        param_init(5) = param_BoundL(5) + (param_BoundU(5)-param_BoundL(5))*(rem(mode.param_seed_indx,4)/4+1/8);
        param_init(6) = param_BoundL(6) + (param_BoundU(6)-param_BoundL(6))*(floor(mode.param_seed_indx/4)/4+1/8);
    else        
        param_init(5)=param_BoundL(5)+(param_BoundU(5)-param_BoundL(5))*rand();
        param_init(6)=param_BoundL(6)+(param_BoundU(6)-param_BoundL(6))*rand();
    end
elseif parameter_indx<281
    mode.USE_FWDSARSA_ONLY=0;      %arbitration
    if ceil((parameter_indx-40)/120)==1 % 0:dual_bayes 1:mixed_albs
        mode.opt_m2=0;
    elseif ceil((parameter_indx-40)/120)==2
        mode.opt_m2=1;
    end
    tmp_parameter_indx=rem((parameter_indx-40),120);
    if tmp_parameter_indx==0
        tmp_parameter_indx=120;
    end
    if ceil(tmp_parameter_indx/60)==1
        mode.num_params=6;
    elseif ceil(tmp_parameter_indx/60)==2
        mode.num_params=8;
    end
    tmp_parameter_indx=rem((parameter_indx-40),60);
    if tmp_parameter_indx==0
        tmp_parameter_indx=60;
    end
    if ceil(tmp_parameter_indx/20)==1
        mode.opt_ArbModel=0; %Arbitration
    elseif ceil(tmp_parameter_indx/20)==2
        mode.opt_ArbModel=2; %Mean_based
    elseif ceil(tmp_parameter_indx/20)==3
        mode.opt_ArbModel=1; %uncertainty based
    end
    mode.param_seed_indx=rem(tmp_parameter_indx,20);
    if mode.param_seed_indx==0
        mode.param_seed_indx=20;
    end
    switch mode.opt_ArbModel
        case 0
            if mode.num_params==8
                param_init=[0.5   0.2    3    3   3    3    0.1    0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
                param_BoundL=[0.1   0.1    1    1   1    1    0.1    0.1];%[0.3   0.05    1    1   1    1    0.15    0.05];% [0.2, 8, 0.8*param_init(3:1:4), 0.12, 0.02];
%                 param_BoundL(7) = 0.5;
                param_BoundU=[0.9   0.9    8    5   10    12    0.9    0.9];%[0.75   0.5    8    5   10    12    0.25    0.15];% [0.6, 12, 1.2*param_init(3:1:4), 0.2, 0.14];
                mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
                if mode.param_seed_indx<17
                    if rem(mode.param_seed_indx,2)==1
                        param_init(ceil((mode.param_seed_indx)/2))=(param_init(ceil((mode.param_seed_indx)/2))+param_BoundL(ceil((mode.param_seed_indx)/2)))/2;
                    else
                        param_init(ceil((mode.param_seed_indx)/2))=(param_init(ceil((mode.param_seed_indx)/2))-param_BoundL(ceil((mode.param_seed_indx)/2)))/2;
                    end
                else
                    change_indx=datasample(1:mode.num_params,randsample(mode.num_params,1));
                    for ii=change_indx
                        param_init(ii)=param_BoundL(ii)+(param_BoundU(ii)-param_BoundL(ii))*rand();
                    end
                end

            elseif mode.num_params==6
                param_init=[0.5,0.2,3,3,0.1,0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
                param_BoundL= [0.1, 0.1, 1,1, 0.1 0.1];%[0.3, 0.05, 1,1 , 0.15, 0.05];
%                 param_BoundL(5) = 0.5;
                param_BoundU= [0.9, 0.9, 8,10, 0.9, 0.9];%[0.75, 0.5, 8,10, 0.25, 0.15];
                mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
                if mode.param_seed_indx<13
                    if rem(mode.param_seed_indx,2)==1
                        param_init(ceil((mode.param_seed_indx)/2))=(param_init(ceil((mode.param_seed_indx)/2))+param_BoundL(ceil((mode.param_seed_indx)/2)))/2;
                    else
                        param_init(ceil((mode.param_seed_indx)/2))=(param_init(ceil((mode.param_seed_indx)/2))-param_BoundL(ceil((mode.param_seed_indx)/2)))/2;
                    end
                else
                    change_indx=datasample(1:mode.num_params,randsample(mode.num_params,1));
                    for ii=change_indx
                        param_init(ii)=param_BoundL(ii)+(param_BoundU(ii)-param_BoundL(ii))*rand();
                    end
                end
            end
        case {1,2} % 3rd, 4th param useless
            if mode.num_params==8
                param_init=[0.5   0.2    3    3   3    3    0.1    0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
                param_BoundL=[0.1   0.1    1    1   1    1    0.1    0.1];%[0.3   0.05    1    1   1    1    0.15    0.05];% [0.2, 8, 0.8*param_init(3:1:4), 0.12, 0.02];
%                 param_BoundL(7) = 0.5;
                param_BoundU=[0.9   0.9    8    5   10    12    0.9    0.9];%[0.75   0.5    8    5   10    12    0.25    0.15];% [0.6, 12, 1.2*param_init(3:1:4), 0.2, 0.14];
                mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
                if mode.param_seed_indx<17
                    tmp_indx = mode.param_seed_indx;
                    param_init(1)=param_BoundL(1) + (param_BoundU(1)-param_BoundL(1))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(2)=param_BoundL(2) + (param_BoundU(2)-param_BoundL(2))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(7)=param_BoundL(7) + (param_BoundU(7)-param_BoundL(7))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(8)=param_BoundL(8) + (param_BoundU(8)-param_BoundL(8))*(rem(tmp_indx,2)/2+1/4);
                else                        
                    for ii=[1,2,5,6]
                        param_init(ii)=param_BoundL(ii)+(param_BoundU(ii)-param_BoundL(ii))*rand();
                    end
                end
            elseif mode.num_params==6
                param_init=[0.5,0.2,3,3,0.1,0.1];%[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];    
                param_BoundL= [0.1, 0.1, 1,1, 0.1 0.1];%[0.3, 0.05, 1,1 , 0.15, 0.05];
%                 param_BoundL(5) = 0.5;
                param_BoundU= [0.9, 0.9, 8,10, 0.9, 0.9];%[0.75, 0.5, 8,10, 0.25, 0.15];
                mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)
                if mode.param_seed_indx<17
                    tmp_indx = mode.param_seed_indx;
                    param_init(1)=param_BoundL(1) + (param_BoundU(1)-param_BoundL(1))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(2)=param_BoundL(2) + (param_BoundU(2)-param_BoundL(2))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(5)=param_BoundL(5) + (param_BoundU(5)-param_BoundL(5))*(rem(tmp_indx,2)/2+1/4);
                    tmp_indx =ceil(tmp_indx)/2;
                    param_init(6)=param_BoundL(6) + (param_BoundU(6)-param_BoundL(6))*(rem(tmp_indx,2)/2+1/4);
                else                        
                    for ii=[1,2,5,6]
                        param_init(ii)=param_BoundL(ii)+(param_BoundU(ii)-param_BoundL(ii))*rand();
                    end
                end
            end
    end
else
    disp('invalid parameter indx')
end
mode.USE_BWDupdate_of_FWDmodel=1; % 1: use the backward update for goal-directed model (fwd model), 0: do not use
mode.DEBUG_Q_VALUE_CHG=0; % Debug option 1: show Q-value before/after whenever there is a goal change.
mode.path_ext=path0;
mode.total_simul=20; % # of total simulation repetition per subject
mode.simul_process_display=0; % 1: display model's process, 0: no diplay
mode.experience_sbj_events=[1, 1];%job_opt.experience_sbj_events; % [pre, main]  +1: experience exactly the same events(decision,state) as subjects. 0: model's own experience -1: use saved setting
mode.max_iter=200; % maximum iteration for optimization
post_filetext=['_sjh' sprintf('%.3d',parameter_indx) '_extended'];
% post_filetext=['_sjh' sprintf('%.3d',parameter_indx) '_diff_range'];
if only_40
    post_filetext = ['_sjh' sprintf('%.3d',parameter_indx) '_40'];
end
mode.task_2019=task_type; %Task type 2019/2020/2021
mode.out=1; % 1: normal evaluation mode, 99: regressor added to the SBJ, 0: debug mode, 2: result scenario specific regressor, 3: high uncertainty fitting
mode.data_type = data_type;

%% OPTION - Regressor arrangement
% {'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd', 'Qarb','uncertaintyM2', 'invFanoM2', 'duncertaintyM1', 'dinvFanoM1', 'weigtM1'};
% should add the regressors in the order of importance
% param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd','Qarb','uncertaintyM2', 'invFanoM2', 'weigtM1'};
param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'MaxinvFano12','dinvFano12','Qsarsa', 'Qfwd', 'Qarb', 'dQbwdEnergy', 'weigtM1', 'invF1_corrected', 'invF2_corrected'};
reg_type_go_first=[1 1.5 2]; % [CAUTION] The order should match with 'param_regressor_type_cue'.   [CAUTION] type"2" should go always last!!!
Do_create_regressors=1;
Is_save_files_local=0; % save optimization parameters and regressor files
Is_save_files_cluster=1; % save optimization parameters and regressor files


%% OPTION - behaviroal analysis & display
% Do_behavioral_analysis=[0];
% if(Do_behavioral_analysis(1)==1) % dont need to create regressors in behavioral analysis mode!
%     Do_create_regressors=0;
% end
% 




%% initialization
if(Is_save_files_local==0)    disp('### files will not be saved to your local PC.');     end
if(Is_save_files_cluster==0)    disp('### files will not be saved to the cluster PC.');     end
    
use_model_regressor_cue=0;  ind_regressor_total=[];   type_regressor=[];    ind_regressor_total_in_design_mat=[];
for ii=1:1:size(param_regressor_type_cue,2) % collect regressor information
    if(strcmp(param_regressor_type_cue{1,ii},'SPE')==1)    use_model_regressor_cue=1;  ind_chk=1;   end
    if(strcmp(param_regressor_type_cue{1,ii},'RPE')==1)    use_model_regressor_cue=1;  ind_chk=2;   end
    if(strcmp(param_regressor_type_cue{1,ii},'uncertaintyM1')==1)    use_model_regressor_cue=1;    ind_chk=3;   end
    if(strcmp(param_regressor_type_cue{1,ii},'uncertaintyM2')==1)    use_model_regressor_cue=1;    ind_chk=4;   end
    if(strcmp(param_regressor_type_cue{1,ii},'meanM1')==1)    use_model_regressor_cue=1;    ind_chk=5;   end
    if(strcmp(param_regressor_type_cue{1,ii},'meanM2')==1)    use_model_regressor_cue=1;    ind_chk=6;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invFanoM1')==1)    use_model_regressor_cue=1;    ind_chk=7;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invFanoM2')==1)    use_model_regressor_cue=1;    ind_chk=8;   end
    if(strcmp(param_regressor_type_cue{1,ii},'weigtM1')==1)    use_model_regressor_cue=1;    ind_chk=9;   end
    if(strcmp(param_regressor_type_cue{1,ii},'weigtM2')==1)    use_model_regressor_cue=1;    ind_chk=10;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qfwd')==1)    use_model_regressor_cue=1;    ind_chk=11;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qsarsa')==1)    use_model_regressor_cue=1;    ind_chk=12;   end
    if(strcmp(param_regressor_type_cue{1,ii},'Qarb')==1)    use_model_regressor_cue=1;    ind_chk=13;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dQbwdEnergy')==1)    use_model_regressor_cue=1;    ind_chk=14;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dQbwdMean')==1)    use_model_regressor_cue=1;    ind_chk=15;   end
    if(strcmp(param_regressor_type_cue{1,ii},'duncertaintyM1')==1)    use_model_regressor_cue=1;    ind_chk=16;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dinvFanoM1')==1)    use_model_regressor_cue=1;    ind_chk=17;   end
    if(strcmp(param_regressor_type_cue{1,ii},'TR_alpha')==1)    use_model_regressor_cue=1;    ind_chk=18;   end
    if(strcmp(param_regressor_type_cue{1,ii},'TR_beta')==1)    use_model_regressor_cue=1;    ind_chk=19;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dinvFano12')==1)    use_model_regressor_cue=1;    ind_chk=20;   end
    if(strcmp(param_regressor_type_cue{1,ii},'ABSdinvFano12')==1)    use_model_regressor_cue=1;    ind_chk=21;   end
    if(strcmp(param_regressor_type_cue{1,ii},'MaxinvFano12')==1)    use_model_regressor_cue=1;    ind_chk=22;   end
    if(strcmp(param_regressor_type_cue{1,ii},'CONFLICTinvFano12')==1)    use_model_regressor_cue=1;    ind_chk=23;   end
    if(strcmp(param_regressor_type_cue{1,ii},'RPE_ABS')==1)    use_model_regressor_cue=1;    ind_chk=24;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invF1_corrected')==1)    use_model_regressor_cue=1;    ind_chk=25;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invF2_corrected')==1)    use_model_regressor_cue=1;    ind_chk=26;   end
    if(strcmp(param_regressor_type_cue{1,ii},'invF1_normalized')==1)    use_model_regressor_cue=1;    ind_chk=27;   end
    if(strcmp(param_regressor_type_cue{1,ii},'dinvFanoCorrected12')==1)    use_model_regressor_cue=1;    ind_chk=28;   end
    
    % index of regressor in "SBJ" structure
    ind_regressor_total=[ind_regressor_total ind_chk];
    % regressor type
    type_regressor=[type_regressor TYPE_REGRESSOR(ind_chk)];
end
% make a regressor index matrix for 1st parametric modulations (normal)
% (1) make 'param_regressor_type_cue_abs_pos_in_design_mat'
reg_cnt=0; param_regressor_type_cue_abs_pos_in_design_mat=[];
ind_regressor_type_base{1,1}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(1)));
reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,1}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [2:1:reg_cnt]];
ind_regressor_type_base{1,2}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(2)));
reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,2}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+2:1:reg_cnt]];
if(reg_type_go_first(3)==1.7)
    ind_regressor_type_base{1,3}.ind_reg=ind_regressor_total(find(type_regressor==reg_type_go_first(3)));
    reg_cnt=reg_cnt+1+length(ind_regressor_type_base{1,3}.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+2:1:reg_cnt]];
end

% make a regressor index for 2nd parametric modulations (dummy)
ind_regressor_type_dummy.ind_reg=ind_regressor_total(find(type_regressor==2));
reg_cnt=reg_cnt+length(ind_regressor_type_dummy.ind_reg);   param_regressor_type_cue_abs_pos_in_design_mat=[param_regressor_type_cue_abs_pos_in_design_mat [param_regressor_type_cue_abs_pos_in_design_mat(end)+1:1:reg_cnt]];
for j=1:1:length(ind_regressor_type_dummy.ind_reg)
    ind_regressor_type_dummy.name{1,j}=LIST_REGRESSOR{1,ind_regressor_type_dummy.ind_reg(j)};
end
% (2)
ind_regressor_type_base{1,1}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(1)));
ind_regressor_type_base{1,2}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(2)));
if(reg_type_go_first(3)==1.7)
    ind_regressor_type_base{1,3}.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==reg_type_go_first(3)));
end
ind_regressor_type_dummy.abs_pos_in_design_mat=param_regressor_type_cue_abs_pos_in_design_mat(find(type_regressor==2));


if(sum(abs(param_regressor_type_cue_abs_pos_in_design_mat-sort(param_regressor_type_cue_abs_pos_in_design_mat,'ascend')))~=0)
    error('- ERROR!!!!: the variable ''param_regressor_type_cue_abs_pos_in_design_mat'' should be in ascending order!!!');
end



%% subject data loading

% which subject to be included
% ### READ ONLY ONE SBJ BECAUSE EACH MODEL WILL LEARN EACH SBJ BEHAVIOR.
ind_sbj_included=list_sbj_included;      SUB_ARRAY=list_sbj_included;
num_sbj_included=length(ind_sbj_included);
ind_included=ind_sbj_included;

for k=1:1:num_sbj_included
    LIST_SBJ_included{1,k}=LIST_SBJ{1,ind_sbj_included(k)};
end
for i=1:1:num_sbj_included %=1. process only 1 subject
    
    SBJ{1,i}.name=LIST_SBJ{1,ind_sbj_included(i)};
    
    % 'pre' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=[LIST_SBJ{1,ind_sbj_included(i)} '_pre_info.mat'];
    if(is_on_cluster==0)
        file_name_full=[mode.path_ext '\result_save\' file_name];
    else
        file_name_full=[mode.path_ext '/result_save/' file_name];
    end
    load(file_name_full);
    if only_40
        for ii = 1:length(HIST_behavior_info{1,1})
            if HIST_behavior_info{1,1}(ii,19) == -1
                if HIST_behavior_info{1,1}(ii,17) ~= 40
                    HIST_behavior_info{1,1}(ii,17) = 0;
                    HIST_map_state_info{1,1}.map(ii).map_sbj(1).reward_save = [0,0,0,0,0,40,0,0,0];
                    HIST_map_state_info{1,1}.map(ii).map_sbj(2).reward_save = [0,0,0,0,0,40,0,0,0];
                end
                if ii == 1
                    HIST_behavior_info{1,1}(ii,18) = HIST_behavior_info{1,1}(ii,17);
                else
                    HIST_behavior_info{1,1}(ii,18) = HIST_behavior_info{1,1}(ii-1,18)+HIST_behavior_info{1,1}(ii,17);
                end
            end
        end
    end
    SBJ{1,i}.HIST_block_condition_pre=HIST_block_condition;
    SBJ{1,i}.HIST_behavior_info_pre=HIST_behavior_info;
    if task_type == 2019 || task_type == 2020
        SBJ{1,i}.HIST_map_state_info_pre=HIST_map_state_info;
    end
    
    % 'fmri' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=[LIST_SBJ{1,ind_sbj_included(i)} '_fmri_info.mat'];
    if(is_on_cluster==0)
        file_name_full=[mode.path_ext '\result_save\' file_name];
    else
        file_name_full=[mode.path_ext '/result_save/' file_name];
    end
    load(file_name_full);
    if only_40
        for sess = 1:length(HIST_behavior_info)
            for ii = 1:length(HIST_behavior_info{1,sess})
                if HIST_behavior_info{1,sess}(ii,19) == -1
                    if HIST_behavior_info{1,sess}(ii,17) ~= 40
                        HIST_behavior_info{1,sess}(ii,17) = 0;
                        HIST_map_state_info{1,sess}.map(ii).map_sbj(1).reward_save = [0,0,0,0,0,40,0,0,0];
                        HIST_map_state_info{1,sess}.map(ii).map_sbj(2).reward_save = [0,0,0,0,0,40,0,0,0];
                    end
                    if ii == 1
                        HIST_behavior_info{1,sess}(ii,18) = HIST_behavior_info{1,1}(ii,17);
                    else
                        HIST_behavior_info{1,sess}(ii,18) = HIST_behavior_info{1,1}(ii-1,18)+HIST_behavior_info{1,1}(ii,17);
                    end
                end
            end
        end
    end
    SBJ{1,i}.HIST_behavior_info=HIST_behavior_info;
    SBJ{1,i}.HIST_behavior_info_Tag=HIST_behavior_info_Tag;
    SBJ{1,i}.HIST_event_info=HIST_event_info;
    SBJ{1,i}.HIST_event_info_Tag=HIST_event_info_Tag;
    SBJ{1,i}.HIST_block_condition=HIST_block_condition;
    SBJ{1,i}.HIST_block_condition_Tag=HIST_block_condition_Tag;
    if task_type == 2019 || task_type == 2020
        SBJ{1,i}.HIST_map_state_info=HIST_map_state_info;
        SBJ{1,i}.HIST_map_state_info_pre_Tag=HIST_map_state_info_Tag;
    end
    num_tot_session=size(SBJ{1,i}.HIST_behavior_info,2);
    
    SBJ{1,i}.map_type=LIST_sbj_map_type(ind_sbj_included(i));
    
    % [fixing part!!! - for Oliver]
    if(strcmp(SBJ{1,i}.name,'Oliver'))
        for mm=1:1:size(SBJ{1,i}.HIST_event_info,2) % each session
            mat_fixing=SBJ{1,i}.HIST_event_info{1,mm};
            index_delete=zeros(1,size(mat_fixing,2));
            [r_fix, c_fix]= find(mat_fixing(7,:)==9);
            for nn=1:1:length(c_fix)
                % check the previous event
                if(mat_fixing(7, c_fix(nn)-1)~=0.5)
                    index_delete(c_fix(nn))=1;
                end
            end
            [tmp c_keep]=find(index_delete==0);
            mat_fixed=mat_fixing(:,c_keep);
            SBJ{1,i}.HIST_event_info{1,mm}=mat_fixed;
        end
    end
    
    
    % [NOTE] now we have 4 variables: mode.HIST_block_condition_pre, mode.HIST_block_condition, mode.HIST_behavior_info_pre, mode.HIST_behavior_info
    % to read a block condition, use "block_condition=mode.HIST_block_condition{1,session_ind}(2,block_ind); % G:1,G':2,H:3,H':4"
    
    %     swsw_amount_pre = [swsw_amount_pre mode.HIST_behavior_info_pre{1,1}(end,17)];
    tot_amount_earned_main_each_sbj =[];
    for jk=1:1:size(SBJ{1,i}.HIST_behavior_info,2)    tot_amount_earned_main_each_sbj = [tot_amount_earned_main_each_sbj; SBJ{1,i}.HIST_behavior_info{1,jk}(end,17)]; end
    %     swsw_amount_main=[swsw_amount_main tot_amount_earned_main_each_sbj];
end







%% model optimization
% param_in(1): myArbitrator.PE_tolerance_m1 (m1's threshold for zero PE)
% param_in(2): m2_absPEestimate_lr % (before) myArbitrator.PE_tolerance_m2 (m2's threshold for zero PE)
% param_in(3): myArbitrator.A_12
% param_in(x): myArbitrator.B_12 : based on A12
% param_in(4): myArbitrator.A_21
% param_in(x): myArbitrator.B_21 : based on A21
% param_in(5): myArbitrator.tau_softmax/param_sarsa.tau/param_fwd.tau : better to fix at 0,2. This should be determined in a way that maintains softmax values in a reasonable scale. Otherwise, this will drive the fitness value!
% param_in(6): % param_sarsa.alpha/param_fwd.alpha 0.01~0.2 to ensure a good "state_fwd.T" in phase 1


% param_init=[0.4995   10.7958    2.0531    4.5944   18.1288    5.0023    0.1645    0.0882];
% param_BoundL=[0.3   5   2.0531    4.5944   18.1288    5.0023   0.15    0.05];% [0.2, 8, 0.8*param_init(3:1:4), 0.12, 0.02];
% param_BoundU=[0.75   15   2.0531    4.5944   18.1288    5.0023   0.25    0.15];% [0.6, 12, 1.2*param_init(3:1:4), 0.2, 0.14];
% mode.boundary_12=0.12;       mode.boundary_21=0.02; % boundary condition : gating fn(1)


mode.param_init=param_init; mode.param_BoundL=param_BoundL; mode.param_BoundU=param_BoundU;
mode.param_length=length(mode.param_init);
% load('\\143.248.30.101\sjh\fmri2022\Behavior_Simul\task_2020\SBJ_structure_sjh_con2_extended.mat')
save_file_name=[path0 '/SBJ_structure_sjh' sprintf('_con%d',ceil(parameter_indx/20)) '_extended.mat'];
load(save_file_name)
param_list = zeros(length(SBJ2),6);
if parameter_indx > 100 && parameter_indx <= 160
    param_list = zeros(length(SBJ2),8);
elseif parameter_indx > 220 && parameter_indx <= 280
    param_list = zeros(length(SBJ2),8);
end
for sbj_indx = 1:length(SBJ2)
    param_list(sbj_indx,:) = SBJ2{1,sbj_indx}.model_BayesArb.param;
end
% ## (way1-each) optimizing for *each* subject and plug the result into each SBJ structure
if(option_optimizing_model==0)
    for ind_sbj=1:1:size(SBJ2,2)
        clear SBJ_test;
        SBJ_test{1,1}=SBJ2{1,ind_sbj};
        disp('############################################')
        disp(['#### optimizing RL-arbitrator for ' sprintf('SBJ#%02d...',ind_sbj)]);
        disp('############################################')
        % [1] model optimization
        mode.out=1;
        myFunc_bu = @(x) eval_ArbitrationRL3_readout(x, SBJ_test, mode); % define a new anonymous function: eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
%         [model_BayesArb.param, model_BayesArb.val]=fminsearchbnd(myFunc_bu, param_init, param_BoundL, param_BoundU, optimset('Display','iter','MaxIter',mode.max_iter));   % X0,LB,UB
        % [2-1] add regressor vector to SBJ
        mode.out=99;
        model_BayesArb.param = param_list(ind_sbj,:);
        SBJ_test=eval_ArbitrationRL3_readout(model_BayesArb.param,SBJ_test,mode); %: eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
        % [3] Save
        model_BayesArb.mode=mode;
%         SBJ_test{1,1}.model_BayesArb=model_BayesArb;
        SBJ2{1,ind_sbj}=SBJ_test{1,1};
        save_file_name=[path0 '/SBJ_structure_sjh' sprintf('_con%d',ceil(parameter_indx/20)) '_extended_readout.mat'];
        if(Is_save_files_local==1)
            eval(['save ' save_file_name ' SBJ2'])
        end
        if(Is_save_files_cluster==1)
            eval(['save ' save_file_name ' SBJ2'])
        end
    end
%     option_optimizing_model=2; % and then write regressors to SBJ structure based on this optimized parameter
end

if(option_optimizing_model==1)
    % ## (way2-batch) optimizing for *all* subjects and plug the result into each SBJ structure
    % [0] retrieve intial configuration for skipping pre-training
%     SBJ_keep=SBJ;
%     load_file_name=['SBJ_structure(backup,batch,Oct30_4).mat'];
%     eval(['load ' save_path_result load_file_name]);
%     for ff1=1:1:length(SBJ_keep)
%         SBJ_keep{1,ff1}.init_state_fwd=SBJ{1,ff1}.init_state_fwd;    SBJ_keep{1,ff1}.init_state_sarsa=SBJ{1,ff1}.init_state_sarsa;
%     end
%     SBJ=SBJ_keep;
    % [1] model optimization
    mode.out=1;
    myFunc_bu = @(x) eval_ArbitrationRL3_readout(x, SBJ, mode); % define a new anonymous function : eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
    bunch=[5 20 40 60 mode.max_iter];
    for mm=[1:1:length(bunch)-1] % split optimization into bunch and save everytime
        max_iter_current=bunch(mm+1)-bunch(mm);
%         [model_BayesArb.param, model_BayesArb.val]=fminsearchbnd(myFunc_bu, param_init, param_BoundL, param_BoundU, optimset('Display','iter','MaxIter',max_iter_current));   % X0,LB,UB
        param_init=model_BayesArb.param;
        % [2-1] add regressor vector to SBJ
        mode.out=99;
        SBJ=eval_ArbitrationRL3_readout(model_BayesArb.param,SBJ,mode); % : eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
        % [3] save
        model_BayesArb.mode=mode;
        for ind_sbj=1:1:size(SBJ,2) % plug in a identical parameter (because this is batch)
%             SBJ{1,ind_sbj}.model_BayesArb=model_BayesArb;
        end
        save_file_name=[path0 '/SBJ_structure' post_filetext '.mat'];
        if(Is_save_files_local==1)
            eval(['save ' save_path_result save_file_name ' SBJ'])
        end
        if(Is_save_files_cluster==1)
            eval(['save ' save_path_result save_file_name ' SBJ'])
        end
    end    
    option_optimizing_model=2; % and then write regressors to SBJ structure based on this optimized parameter
end

if(option_optimizing_model==2) % [NOTE] replace initial SBJ = just read SBJ from the "SBJ_structure.mat"
    load_file_name=['SBJ_structure_sjh.mat'];
    eval(['load ' save_path_result load_file_name])
    % regressor part deleting and regenerating.
    mode.param_length=6;
    for ff=1:1:length(list_sbj_included)
        disp(sprintf('- writing regressor to SBJ structure (SBJ%02d)...',list_sbj_included(ff)));
        % find my subject in "SBJ" strucure of the 'SBJ_structure.mat' file        
        did_find=0;
        for ss=1:1:size(SBJ,2)
            if(strcmp(SBJ{1,ss}.name,LIST_SBJ_included{1,ff})==1)
                SBJ0{1,1}=SBJ{1,ss}; % SBJ : this includes SBJ structure for subjects to be included for this code
                did_find=did_find+1;
            end
        end
        if(did_find~=1)            error('-ERROR:: no correponding subject found in the "SBJ_structure.mat" file!!!');   end
        
        if(isfield(SBJ0{1,1}, 'regressor')==1)
            SBJ0{1,1}=rmfield(SBJ0{1,1},'regressor'); %remove the regressor field
        end
        mode.out=99;        
        model_BayesArb.param=SBJ0{1,1}.model_BayesArb.param;
        SBJ0=eval_ArbitrationRL3_readout(model_BayesArb.param,SBJ0,mode); % refresh and add the regressor part : eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
        SBJ1{1,ff}=SBJ0{1,1};
    end
    clear SBJ
    SBJ=SBJ1;
    if(update_SBJ_structure==1)        eval(['save ' save_path_result load_file_name ' SBJ']);  end
end

if(option_optimizing_model==3) % [NOTE] replace initial SBJ = just read SBJ from the "SBJ_structure.mat"
    load_file_name=['SBJ_structure_sjh.mat'];
    load_file_name2=['param.mat'];
    eval(['load ' save_path_result load_file_name])
    eval(['load ' save_path_result load_file_name2])
    % regressor part deleting and regenerating.
    for ff=1:1:length(list_sbj_included)
        disp(sprintf('- writing regressor to SBJ structure (SBJ%02d)...',list_sbj_included(ff)));
        % find my subject in "SBJ" strucure of the 'SBJ_structure.mat' file        
        did_find=0;
        for ss=1:1:size(SBJ,2)
            if(strcmp(SBJ{1,ss}.name,LIST_SBJ_included{1,ff})==1)
                SBJ0{1,1}=SBJ{1,ss}; % SBJ : this includes SBJ structure for subjects to be included for this code
                did_find=did_find+1;
            end
        end
        if(did_find~=1)            error('-ERROR:: no correponding subject found in the "SBJ_structure.mat" file!!!');   end
        
        if(isfield(SBJ0{1,1}, 'regressor')==1)
            SBJ0{1,1}=rmfield(SBJ0{1,1},'regressor'); %remove the regressor field
        end
        mode.out=99;        
        model_BayesArb.param=param_model(list_sbj_included,:);
        SBJ0=eval_ArbitrationRL3_readout(model_BayesArb.param,SBJ0,mode); % refresh and add the regressor part : eval_ArbitrationRL2(x, SBJ_test, mode) for full BayesArb
        SBJ1{1,ff}=SBJ0{1,1};
    end
    clear SBJ
    SBJ=SBJ1;
    if(update_SBJ_structure==1)        eval(['save ' save_path_result load_file_name ' SBJ']);  end
end

disp('- all done.')

out=1;
end