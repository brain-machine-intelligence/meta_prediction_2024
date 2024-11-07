% Load the parameter files from the optimization folder, and then test the
% best model.


clear all
close all


%% data collection loop
ZZZ=[];
swsw_score=[];
swsw_amount_pre=[];
swsw_amount_main=[];
for swsw=[1]%[1:1:11]
%%
    
    

%% parameter

EXP_NAME0='GH_';

% behaviroal data
LIST_SBJ={'david', 'DeDe', 'rosemary', 'Boyu', 'melissa', 'Rehevolew', 'joel', 'clarke', 'angela', 'william', 'josephine'}; % (good in pre which is mostly habitual - rosemary, melissa)
% behavioral + fmri data
% LIST_SBJ={'Oliver', 'Hao', 'Breanna', 'Derek', 'Timothy', 'Teagan', 'Jeffrey', 'Seung', 'Carole', 'Tony', 'Surendra', 'Lark'}; 
opt.ind_sbj_included=[swsw];

total_simul=1; % # of total simulation repetition per subject

%etc
NUM_FILE_TO_PROCESS_FOR_EACH_SBJ=total_simul; %  # of simulations of each model for each subject.
% num_repeat_simulation=1; % # of repetition for each model. fixed. do not change.
    

%% simulation options
USE_BWDupdate_of_FWDmodel=1; % 1: use the backward update for goal-directed model (fwd model), 0: do not use
USE_FWDSARSA_ONLY=1; % 0: arbitration, 1: use fwd only, 2: use sarsa only

% Debug option
DEBUG_Q_VALUE_CHG=0; % 1: show Q-value before/after whenever there is a goal change.

%% option parameters
opt.SIMUL_PC=1; % 1:local   2:gpu
if(opt.SIMUL_PC==1)
    opt.str_path_div='\';   addpath('/home/swlee/simul/Model_RL');
else
    opt.str_path_div='/';
end
% 0.experiment
opt.exp_name0=[1];
opt.TARGET_FOLDER_NAME='result_simul';
opt.ind_included=[1];
opt.EXP_NAME=['ANY_start_' sprintf('%02d_Sbj%02d',opt.exp_name0,opt.ind_included)];
if(opt.SIMUL_PC==1)
    opt.path_ext=['F:' opt.str_path_div '0-Program' opt.str_path_div 'MATLABwork' opt.str_path_div 'work' opt.str_path_div 'ModelRL'];
else
    opt.path_ext=['/home/swlee/simul/ModelRL'];
end
mkdir([opt.path_ext opt.str_path_div opt.TARGET_FOLDER_NAME],opt.EXP_NAME);



%% Devaluation/coningency degradation/risk schedule
% (1) devaluation schedule
pt_devaluation=[1000];%[41:1:80];%[1:1:40]; % second phase 1~80. if devaluation point > num_max_trial, then no devaluation
pt_ref=pt_devaluation(1);
% (2) degradation schedule
pt_degradation=[1000];
% (3) transition prob schedule
pt_prob_chg_max_risk=[1000];%[11:1:40]; % second phase 1~80.
pt_prob_chg_min_risk=[1000];%[41:1:80]; %[41:1:80]; % second phase 1~80.
pt_prob_chg_reversed=[1000]; % second phase 1~80.
% (4) reward prob schedule
pt_prob_rwd_chg=[1000];
% (5) goal-directed mode (keep changing rwd values)
pt_devaluation_goal=[1000];%[31:1:60 91:1:120 151:1:180];
rwd_aversion_factor=0; % 1: no change, 0: zero reward (ok), -1:opposite (best)
% scenario 3
% ss0=[1:1:30]; ss1=[31:1:60]; ss2=[61:1:90]; ss3=[91:1:120];   ss4=[121:1:150];   ss5=[151:1:180];
% pt_prob_chg_max_risk=[ss2 ss4];
% pt_prob_chg_min_risk=[ss1 ss3 ss5];
% pt_devaluation_goal=[ss1 ss3 ss5];


%% variables
X_all=[]; % behavior profile vectors
Y_all=[]; % [parameter vector, subject index]
X_in_fwd=[];  X_in_sarsa=[];
X_in_fwd_mean=[];  X_in_sarsa_mean=[];
X_in_fwd_var=[];  X_in_sarsa_var=[];
HIST_best_SumNegLogLik=[];
% collection_behavior_profile=[];

%% data variable to which all the bevahior are to be saved
% variables:: data_behavior, data_model
data_behavior=cell(1,NUM_FILE_TO_PROCESS_FOR_EACH_SBJ);
data_model.pt_prob_chg_max_risk=pt_prob_chg_max_risk;
data_model.pt_prob_chg_min_risk=pt_prob_chg_min_risk;
data_model.pt_devaluation_goal=pt_devaluation_goal;

% base path
%     Opt.path_ext=['F:' Opt.str_path_div '0-Program' Opt.str_path_div 'MATLABwork' Opt.str_path_div 'work' Opt.str_path_div 'ModelRL'];
opt.path_ext=pwd;


%% subject data loading (good in pre - rosemary, melissa)
pop_all=[1:1:size(LIST_SBJ,2)];
pop_non_caltech=[1 2 4:8 11]; pop_caltech=[3 9 10]; pop_male=[1 3 7 8 10]; pop_female=[2 4 5 6 9 11];   pop_asian=[2 3 6 9 10 11];  pop_caucasian=[1 4 5 7 8];
num_sbj_total=size(LIST_SBJ,2);

%% which subject to be included
% ### READ ONLY ONE SBJ BECAUSE EACH MODEL WILL LEARN EACH SBJ BEHAVIOR.
ind_sbj_included=opt.ind_sbj_included;      SUB_ARRAY=opt.ind_sbj_included;
num_sbj_included=length(ind_sbj_included);
ind_included=ind_sbj_included;
for k=1:1:num_sbj_included
    LIST_SBJ_included{1,k}=LIST_SBJ{1,ind_sbj_included(k)};
end
for i=1:1:num_sbj_included %=1. process only 1 subject
    
    % 'pre' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=[LIST_SBJ{1,ind_sbj_included(i)} '_pre_info.mat'];
    file_name_full=[opt.path_ext '\result_save\' file_name];
    load(file_name_full);
    HIST_block_condition_pre=HIST_block_condition;
    HIST_behavior_info_pre=HIST_behavior_info;
        
    % 'fmri' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
    file_name=[LIST_SBJ{1,ind_sbj_included(i)} '_fmri_info.mat'];
    file_name_full=[opt.path_ext '\result_save\' file_name];
    load(file_name_full);    
    num_tot_session=size(HIST_behavior_info,2);
        
    % [NOTE] now we have 4 variables: HIST_block_condition_pre, HIST_block_condition, HIST_behavior_info_pre, HIST_behavior_info
    % to read a block condition, use "block_condition=HIST_block_condition{1,session_ind}(2,block_ind); % G:1,G':2,H:3,H':4"    
    
    swsw_amount_pre = [swsw_amount_pre HIST_behavior_info_pre{1,1}(end,17)];
    tot_amount_earned_main_each_sbj =[];
    for jk=1:1:5    tot_amount_earned_main_each_sbj = [tot_amount_earned_main_each_sbj; HIST_behavior_info{1,jk}(end,17)]; end
    swsw_amount_main=[swsw_amount_main tot_amount_earned_main_each_sbj];
end












%% create MAP
map_opt.transition_prob_seed=[0.5 0.5];
map_opt.reward_seed=[40 20 10 0];
[myMap N_state N_action N_transition]=Model_Map_Init2('sangwan2012',map_opt);


%% create my arbitrator
myArbitrator=Bayesian_Arbitration_Init(N_state,N_action,N_transition);

%% create my RL
myState=Model_RL_Init(N_state,N_action,N_transition);



%% model parameter - for the functional mode of RL
% SARSA model
param_sarsa.gamma=1.0; % fixed - not actual parameter
%     param_sarsa.alpha=0.15; % learning rate (0.1~0.2)
%     param_sarsa.tau=0.5; % decision focus parameter
% FWD model (for the latent stage)
% param_fwd0.alpha=0.1; % learning rate
%     param_fwd0.tau=param_sarsa.tau; % decision focus parameter
% FWD model
%     param_fwd.alpha=0.15; % learning rate
%     param_fwd.tau=param_sarsa.tau; % decision focus parameter


% %% [SW] Load behaviour data (pseudo data)
% state_column=[1 2 3];   action_column=[4 5];
% % eval(['load ',[opt.path_ext opt.str_path_div 'behavior_data_glascher' opt.str_path_div 'data.mat']]);
% eval(['load ',[opt.path_ext opt.str_path_div 'behavior_data_pseudo' opt.str_path_div 'data_save.mat']]);
% % and attach it to the map
% ind_included=opt.ind_included;
% if(opt.ind_included(1)==0) % for glascher dataset only
%     ind_included=[1:1:20];
% end
% data0=cell(1,length(ind_included));
% for j=1:1:length(ind_included)
%     data0{1,j}=data_behavior{1,ind_included(j)}.data;
% end
% myMap.data=data0;



%note: Time_Step: 1e-1 (fast) ~ 1e-2 (slow)
parameter_converted=[0.7,1e-2,2.17142857142857,2.77777777777778,1.88888888888889,27.3015873015873,0.12,1,0.100000000000000,2,10];
parameter_converted(7)=0.05; % 0.01~0.2 to ensure a good "state_fwd.T" in phase 1
parameter_converted(9)=0.2;  % better to fix at 0,2. This should be determined in a way that maintains softmax values in a reasonable scale. Otherwise, this will drive the fitness value!




%% variables
X_all=[]; % behavior profile vectors
Y_all=[]; % [parameter vector, subject index]
X_in_fwd=[];  X_in_sarsa=[];
X_in_fwd_mean=[];  X_in_sarsa_mean=[];
X_in_fwd_var=[];  X_in_sarsa_var=[];
HIST_best_SumNegLogLik=[];
% collection_behavior_profile=[];


%% SIMULATION
opt.generation_id=1;
for pop_id=1:1:1
    
    
    disp(sprintf('  :: processing %d-th generation, %d-th population...',opt.generation_id,pop_id));
    
    
    %% [SW] Parameter insertion/change
    Sum_NegLogLik=0.0;
    % arbitrator
    myArbitrator.PE_tolerance_m1=parameter_converted(pop_id,1); % defines threshold for zero PE
    myArbitrator.PE_tolerance_m2=parameter_converted(pop_id,11); % defines threshold for zero PE
    myArbitrator.Time_Step=parameter_converted(pop_id,2); % the smaller, the slower
    myArbitrator.A_12=parameter_converted(pop_id,3);  myArbitrator.B_12=parameter_converted(pop_id,4);
    myArbitrator.A_21=parameter_converted(pop_id,5);  myArbitrator.B_21=parameter_converted(pop_id,6);
    myArbitrator.p=parameter_converted(pop_id,8);
    % SARSA
    param_sarsa.alpha=parameter_converted(pop_id,7); % learning rate (0.1~0.2)
    % FWD
    param_fwd.alpha=parameter_converted(pop_id,7);
    % pop_id=8 takes no effect.
    % Softmax parameter for all models
    myArbitrator.tau_softmax=parameter_converted(pop_id,9); % use the same value as sarsa/fwd
    param_sarsa.tau=parameter_converted(pop_id,9);
    param_fwd.tau=parameter_converted(pop_id,9);
    % arbitrator start
    myArbitrator.ind_active_model=parameter_converted(pop_id,10);
    if(myArbitrator.ind_active_model==1)
        myArbitrator.m1_prob_prev=0.7; % do not use 0.5 which causes oscillation.
        myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
    else
        myArbitrator.m1_prob_prev=0.3;
        myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
    end
    
    
    %% Devaluation/coningency degradation/risk schedule    
    % (1) devaluation schedule
    pt_devaluation=[1000];%[41:1:80];%[1:1:40]; % second phase 1~80. if devaluation point > num_max_trial, then no devaluation
    pt_ref=pt_devaluation(1);
    % (2) degradation schedule
    pt_degradation=[1000];
    % (3) transition prob schedule
    pt_prob_chg_max_risk=[1000];%[1:1:30 61:1:90 121:1:150];%[11:1:40]; % second phase 1~80.
    pt_prob_chg_min_risk=[1000];%[31:1:60 91:1:120 151:1:180];%[41:1:80]; %[41:1:80]; % second phase 1~80.
    pt_prob_chg_reversed=[1000]; % second phase 1~80.
    % (4) reward prob schedule
    pt_prob_rwd_chg=[1000];
    % (5) goal-directed mode (keep changing rwd values)
    pt_devaluation_goal=[1000];%[31:1:60 91:1:120 151:1:180];
    rwd_aversion_factor=0; % 1: no change, 0: zero reward (ok), -1:opposite (best)
    
    % scenario 3
    ss0=[1:1:30]; ss1=[31:1:60]; ss2=[61:1:90]; ss3=[91:1:120];   ss4=[121:1:150];   ss5=[151:1:180];
    pt_prob_chg_max_risk=[ss0 ss2 ss4];
    pt_prob_chg_min_risk=[ss1 ss3 ss5];
    pt_devaluation_goal=[ss1 ss3 ss5];
    
    
    
    for ind_dev=[1:1:length(pt_devaluation)]
        
        %% Multiple simulations
        win_fwd=0; win_sarsa=0;
        
        for ll=1:1:length(ind_included) % each subject
            
            swsw_score_each_simul=[];   swsw_score_each_simul_each_mode=[];
            for kk=1:1:total_simul
                
                Sum_NegLogLik_each_simul=0;
                Sum_NegLogLik_each_simul_eachMode=zeros(1,4);
                Num_occur_eachMode=zeros(1,4); % # of each mode (G,G',H,H')
                OBS=[];

                
                
                %% Simulation
                % (1) phase 1 - random action, no reward
                state_fwd=myState;  state_fwd.name='fwd';
                state_sarsa=myState;    state_sarsa.name='sarsa';
                
                map=myMap;  map0=myMap; map0_s=myMap;
                                
                
                %% (1) phase 1 - 'pre' training
                                          
                num_max_trial0=size(HIST_behavior_info_pre{1,1},1);
                map0.epoch=kk;                map0_s.epoch=kk;                
                map0.data=HIST_behavior_info_pre{1,1};  map0_s.data=HIST_behavior_info_pre{1,1};
                
                opt_state_space.use_data=0; % do not use subject data
                if(1) 
                    % fwd learning
                    i=0;  cond=1;      
                    while ((i<num_max_trial0)&&(cond))
                        i=i+1;
                        map0.trial=i;
                        %             disp(sprintf('- [%d/%d] session...',i,num_max_trial0));
                        % initializing the state
                        [state_fwd map0]=StateClear(state_fwd,map0);
                        while (~state_fwd.JobComplete)                            
                            % decision
%                             state_fwd=Model_RL(state_fwd, param_fwd, map0, 'decision_behavior_data_save');
                            state_fwd=Model_RL(state_fwd, param_fwd, map0, 'decision_random');
                            % state transition
                            [state_fwd map0]=StateSpace_v1(state_fwd,map0,opt_state_space);  % map&state index ++                            
                            % 1. fwd model update
                            state_fwd=Model_RL(state_fwd, param_fwd, map0, 'fwd_update');                           
                        end                        
                    end
                    % sarsa learning
                    i=0;  cond=1;      
                    while ((i<num_max_trial0)&&(cond))
                        i=i+1;
                        map0_s.trial=i;
                        %             disp(sprintf('- [%d/%d] session...',i,num_max_trial0));
                        % initializing the state
                        [state_sarsa map0_s]=StateClear(state_sarsa,map0_s);
                        while (~state_sarsa.JobComplete)
                            % 0. current action selection : (s,a) - using arbitrator's Q-value
                            state_sarsa=Model_RL(state_sarsa, param_sarsa, map0_s, 'decision_random');
                            % 1. sarsa state update (get reward and next state) : (r,s')
                            [state_sarsa map0_s]=StateSpace_v1(state_sarsa,map0_s,opt_state_space); % map&state index ++
                            % 1. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                            state_sarsa=Model_RL(state_sarsa, param_sarsa, map0_s, 'decision_hypo');                            
                            % 1. sarsa model upate
                            state_sarsa=Model_RL(state_sarsa, param_sarsa, map0_s, 'sarsa_update');
                        end  
                    end
%                     % Q-value synchronization
%                     state_sarsa.Q=state_fwd.Q;

                end
         
                disp('- pretraining completed.')
                
                %% (2) phase 2 - intact action, rewards given
                
                num_max_session=size(HIST_behavior_info,2);
                
                cond=1;
                myArbitrator_top=myArbitrator;                
                mode_data_prev=6;
                block_cond_prev=-1; on_h=-1; on_g=-1;
                
                for ind_sess=1:1:num_max_session
                    
                    % enter each session data into map
                    i=0;
                    num_max_trial=size(HIST_behavior_info{1,ind_sess},1);
                    map.epoch=kk;
                    map.data=HIST_behavior_info{1,ind_sess};
                    
                    disp(sprintf('- Sbj[%d], Simul[%d/%d], Session [%d/%d]...',swsw,kk,total_simul,ind_sess,num_max_session));
                                        
                    while ((i<num_max_trial)&&(cond))
                        i=i+1;
                        map.trial=i;
%                         disp(sprintf('- Simul[%d/%d], Session [%d/%d], Trial [%d/%d]...',kk,total_simul,ind_sess,num_max_session,i,num_max_trial));
                        
                        % initializing the state
                        [state_fwd map]=StateClear(state_fwd,map);
                        [state_sarsa map]=StateClear(state_sarsa,map);
                        
                        % read a mode from data
                        mode_data=map.data(map.trial,18);
                        
                        % mode change whenever there is a visual cue change
                        case_number=0;
                        myArbitrator_top.backward_flag=0;
                        if(mode_data_prev~=mode_data) % if there is any visual cue change, then goal mode
                            myArbitrator_top.backward_flag=1; % after backward update, should set it to 0.
                            myArbitrator_top.ind_active_model=1; % switching the mode
                            myArbitrator_top.m1_prob_prev=0.9; % changing the choice prob accordingly
                            myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                            if((mode_data_prev~=-1)&&(mode_data==-1)) % goal mode -> habitual mode
                                myArbitrator_top.ind_active_model=2; % switching the mode
                                myArbitrator_top.m1_prob_prev=0.1; % changing the choice prob accordingly
                                myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                            end
                        end
                        if(USE_FWDSARSA_ONLY==1) % forward only
                            myArbitrator_top.ind_active_model=1; % switching the mode
                            myArbitrator_top.m1_prob_prev=0.99; % changing the choice prob accordingly
                            myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                            myArbitrator_top.Time_Step=1e-2; % extremely slow, so do not switch to the other learner
                        end
                        if(USE_FWDSARSA_ONLY==2) % sarsa only
                            myArbitrator_top.ind_active_model=2; % switching the mode
                            myArbitrator_top.m1_prob_prev=0.01; % changing the choice prob accordingly
                            myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                            myArbitrator_top.Time_Step=1e-2; % extremely slow, so do not switch to the other learner
                        end
                        
                        OBS=[OBS [myArbitrator_top.ind_active_model; mode_data; myArbitrator_top.m1_prob]];
                        %
                        
                        opt_state_space.use_data=0; % do not use subject data for state-transition
                                                                        
                        while (((myArbitrator_top.ind_active_model==1)&&(~map.JobComplete))||((myArbitrator_top.ind_active_model==2)&&(~map.JobComplete)))
                            
                                                        
                            % index synchronization
                            state_fwd.index=map.index;                state_sarsa.index=map.index;
                            
                            
                            %% fwd mode: backward update of fwd model
                            if(myArbitrator_top.backward_flag==1) % if the agent detects context change
                                % (1) revaluation
                                if(mode_data==-1) % reevaluation for habitual mode
                                    mode_data_mat=[6 7 8 9];
                                else % reevaluation for goal mode
                                    mode_data_mat=mode_data;
                                end
                                map.reward=zeros(N_state,1);    map.reward(mode_data_mat)=map.reward_save(mode_data_mat);
                                % (2) backward update of the fwd model
                                if(USE_BWDupdate_of_FWDmodel==1)
                                    state_fwd=Model_RL(state_fwd, param_fwd, map, 'bwd_update');
                                end
                            else
                                % preserve reward values
                                map.reward=map.reward_save;
                            end
                            
                            %% Compute negative log-likelihood : evaluate the arbitrator softmax using sbj's state,action                   
                            state_data=map.data(map.trial,3+map.index);
                            action_data=map.data(map.trial,6+map.index);   
                            % compute real Q-value by merging two Q(fwd,sarsa)
                            myArbitrator_top.Q=...
                            ((myArbitrator_top.m1_prob*state_fwd.Q).^myArbitrator_top.p+...
                            (myArbitrator_top.m2_prob*state_sarsa.Q).^myArbitrator_top.p).^(1/myArbitrator_top.p);
                            var_exp=exp(myArbitrator_top.tau_softmax*myArbitrator_top.Q(state_data,:)); % (N_actionx1)
                            eval_num=log(var_exp(action_data)/sum(var_exp));
                            Sum_NegLogLik=Sum_NegLogLik-eval_num/total_simul;                            
                            Sum_NegLogLik_each_simul=Sum_NegLogLik_each_simul-eval_num;                            
                            % block condition of each trial: 1:G(low T uncertainty), 2:G'(high T uncertainty), 3:H(high T uncertainty), 4:H'(low T uncertainty)
                            block_cond=HIST_block_condition{1,ind_sess}(2,map.trial);                            
                            Sum_NegLogLik_each_simul_eachMode(block_cond)=Sum_NegLogLik_each_simul_eachMode(block_cond)-eval_num/total_simul;
                            Num_occur_eachMode(block_cond)=Num_occur_eachMode(block_cond)+1;                            
                                                        
%                             if((block_cond_prev==4)&&(block_cond==3))
%                                 on_h=1;
%                                 if(action_data~=2)
%                                     err_indicator
%                                 end
%                             end
% %                             if((block_cond_prev==1)&&(block_cond==2))
% %                                 on_g=1;
% %                             end                            
%                             if((block_cond_prev==3)&&(block_cond~=3))
%                                 on_h=0;
%                             end
% %                             if((block_cond_prev==2)&&(block_cond~=2))
% %                                 on_g=0;
% %                             end
%                             block_cond_prev=block_cond;
                            
                            
                            %
%                             ZZZ=[ZZZ; [state_data, action_data, var_exp, (-1)*eval_num]];
                            
                            
                            
                            %% main computation
                            QQ_prev=state_fwd.Q;
                            if(myArbitrator_top.ind_active_model==1) % fwd                                
                                % 0. current action selection : (s,a) - using arbitrator's Q-value
                                state_fwd=Model_RL(state_fwd, param_fwd, map, 'decision_arbitrator', myArbitrator_top, state_sarsa);
%                                 state_fwd=Model_RL(state_fwd, param_fwd, map, 'decision_behavior_data_save');
                                % 1. fwd state update (get reward and next state) : (r,s')                                
                                [state_fwd map]=StateSpace_v1(state_fwd,map,opt_state_space); % map&state index ++
                                % 1. fwd model update                                
                                state_fwd=Model_RL(state_fwd, param_fwd, map, 'fwd_update');
                                % 2. state synchronization
                                state_sarsa.state_history(state_fwd.index)=state_fwd.state_history(state_fwd.index);
                                state_sarsa.SARSA=state_fwd.SARSA;
                                % 3. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                                state_sarsa=Model_RL(state_sarsa, param_sarsa, map, 'decision_hypo');
                                % 3. sarsa model upate
                                state_sarsa=Model_RL(state_sarsa, param_sarsa, map, 'sarsa_update');
                                % history synchronization
                                myArbitrator_top.state_history=state_fwd.state_history;
                                myArbitrator_top.reward_history=state_fwd.reward_history;
                            end
                            QQ_current=state_fwd.Q;
                            
                            if(myArbitrator_top.ind_active_model==2) % sarsa
                                % 0. current action selection : (s,a) - using arbitrator's Q-value
                                state_sarsa=Model_RL(state_sarsa, param_sarsa, map, 'decision_arbitrator', myArbitrator_top, state_fwd);
                                % 1. sarsa state update (get reward and next state) : (r,s')                                
                                [state_sarsa map]=StateSpace_v1(state_sarsa,map,opt_state_space); % map&state index ++
                                % 1. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                                state_sarsa=Model_RL(state_sarsa, param_sarsa, map, 'decision_hypo');
                                % 1. sarsa model upate
                                state_sarsa=Model_RL(state_sarsa, param_sarsa, map, 'sarsa_update');
                                % 2. state synchronization
                                state_fwd.state_history(state_sarsa.index)=state_sarsa.state_history(state_sarsa.index);
                                state_fwd.SARSA=state_sarsa.SARSA;
                                % 3. fwd model update
                                state_fwd=Model_RL(state_fwd, param_fwd, map, 'fwd_update');
                                % history synchronization
                                myArbitrator_top.state_history=state_sarsa.state_history;
                                myArbitrator_top.reward_history=state_sarsa.reward_history;
                            end
                            
                            % ARBITRATOR: transition rate of the sarsa only
                            [myArbitrator_top, state_fwd, state_sarsa]=Bayesian_Arbitration_v3(myArbitrator_top, state_fwd, state_sarsa, map);
                            
                                                        
                            % DISPLAY Q-VAL change according to goal change
                            if(DEBUG_Q_VALUE_CHG==1)
                                disp(sprintf('goal: %d->%d, Q_debug=[state|| fwdQ(L)|fwdQ(R) -> fwdQ(L)|fwdQ(R) | arbQ(L)|arbQ(R)]',mode_data_prev, mode_data))
%                                 Q_debug=[[1:1:9]' QQ_prev QQ_current myArbitrator_top.Q(:,:)]
%                                 myArbitrator_top.m1_prob
%                                 state_fwd.SARSA
                                [state_data, action_data, var_exp, (-1)*eval_num]
                                input('pause...')                                
                            end
                            
                        end
                        
                        % save to previous mode data
                        mode_data_prev=mode_data;
                        
                    end
                    
                    
                    
                end
                
               
                swsw_score_each_simul=[swsw_score_each_simul; Sum_NegLogLik_each_simul];                
                swsw_score_each_simul_each_mode=[swsw_score_each_simul_each_mode; Sum_NegLogLik_each_simul_eachMode./Num_occur_eachMode];
                            
            end
        end
        
        
    end
    
    
    % fitness values (to be maximized)
    fitness_val(pop_id)=1e6*1.0/Sum_NegLogLik;
    %     fitness_val(pop_id)=(-1.0)*Sum_NegLogLik;
    Sum_NegLogLik_val(pop_id)=Sum_NegLogLik;
    disp(sprintf('    = fitness value: %04.4f (NegLogLik: %f)',max(fitness_val),min(Sum_NegLogLik)));
    
end

%% file save
file_name=sprintf('generation%03d_bestfit(%04.4f).mat',opt.generation_id,max(fitness_val));
eval(['save ',[opt.path_ext opt.str_path_div opt.TARGET_FOLDER_NAME opt.str_path_div opt.EXP_NAME opt.str_path_div file_name ' fitness_val Sum_NegLogLik_val parameter_converted opt' ]]);




    
    


%% figure - alpha/beta
if(0)
    
    figure('Name','transition rate and winner')
    ind_line=[1:1:num_max_trial];
    
    subplot(3,1,1)
    plot(ind_line,mean(HIST_TRANSITION_RATE_SARSA,1),'b-',ind_line,mean(HIST_TRANSITION_RATE_FWD,1),'r-')
    title('transition rate');
    legend('SARSA->FWD (sarsa disbelief)','FWD->SARSA (fwd disbelief)')
    
    subplot(3,1,2)
    winner_fwd=mean(HIST_TRANSITION_RATE_SARSA,1)-mean(HIST_TRANSITION_RATE_FWD,1);
    winner_prob=mean(HIST_TRANSITION_RATE_SARSA-HIST_TRANSITION_RATE_FWD,1);
    plot(ind_line,winner_prob);
    title('transition rate difference (FWD>SARSA)');
    
    subplot(3,1,3)
    plot(ind_line,mean(HIST_PROB_CHOICE_FWD,1),'b-',ind_line,mean(HIST_PROB_CHOICE_SARSA,1),'r-')
    title('prob of choosing fwd');
    legend('p(FWD choice)','p(SARSA choice)')
    
    figure('Name','means (line) with variances (shade)')
    str_title{1}='negative PE';     str_title{2}='zero PE';     str_title{3}='positive PE';
    for j=1:1:myArbitrator.K
        subplot(myArbitrator.K,1,j);
        shade_mat_sarsa=reshape(HIST_PE_VAR_SARSA(j,:)',[size(HIST_PE_VAR_SARSA(j,:)',1) 1 size(HIST_PE_VAR_SARSA(j,:)',2)]);
        shade_mat_fwd=reshape(HIST_PE_VAR_FWD(j,:)',[size(HIST_PE_VAR_FWD(j,:)',1) 1 size(HIST_PE_VAR_FWD(j,:)',2)]);
        boundedline(ind_line,HIST_PE_MEAN_SARSA(j,:),shade_mat_sarsa,'alpha','b',ind_line,HIST_PE_MEAN_FWD(j,:),shade_mat_fwd,'alpha','r'); % with 'alpha' option you cannot make a proper screen capture.
        legend('E_{sarsa}(\theta)','E_{fwd}(\theta)');
        axis([min(ind_line) max(ind_line) 0 1])
        title(str_title{j});
    end
    
    figure('Name','inverse Fano')
    str_title{1}='negative PE';     str_title{2}='zero PE';     str_title{3}='positive PE';
    for j=1:1:myArbitrator.K
        subplot(myArbitrator.K,1,j);
        plot(ind_line,HIST_invFano_SARSA(j,:),'b',ind_line,HIST_invFano_FWD(j,:),'r');
        legend('invFano(SARSA)','invFano(FWD)');
        title(str_title{j});
    end
    
end





%% figure

% figure('Name','SARSA model reward')
% rwd_acc=[];
% for i=100:10:length(HIST_RWD_SARSA)
%     rwd_acc=[rwd_acc sum(HIST_RWD_SARSA(i-99:i))];
% end
% plot(rwd_acc);
%
% figure('Name','FWD model reward')
% rwd_acc2=[];
% for i=100:10:length(HIST_RWD_SARSA)
%     rwd_acc2=[rwd_acc2 sum(HIST_RWD_FWD(i-99:i))];
% end
% plot(rwd_acc2);

if(0)
    
    %% t-test
    
    figure('Name','average actions - non-devaluation vs. devaluation (overtraining)')
    
    subplot(2,2,1)
    boxplot(response_rate_distal_sarsa)
    [h,p] = ttest(response_rate_distal_sarsa(:,2),mean(response_rate_distal_sarsa(:,1)));
    str=sprintf('sarsa - distal (t-test p=%e)',p);
    title(str)
    
    subplot(2,2,2)
    boxplot(response_rate_proximal_sarsa)
    [h,p] = ttest(response_rate_proximal_sarsa(:,2),mean(response_rate_proximal_sarsa(:,1)));
    str=sprintf('sarsa - proximal (t-test p=%e)',p);
    title(str)
    
    subplot(2,2,3)
    boxplot(response_rate_distal_fwd)
    [h,p] = ttest(response_rate_distal_fwd(:,2),mean(response_rate_distal_fwd(:,1)));
    str=sprintf('fwd - distal (t-test p=%e)',p);
    title(str)
    
    subplot(2,2,4)
    boxplot(response_rate_proximal_fwd)
    [h,p] = ttest(response_rate_proximal_fwd(:,2),mean(response_rate_proximal_fwd(:,1)));
    str=sprintf('fwd - proximal (t-test p=%e)',p);
    title(str)
    
    %% one way anova table display (similar to t-test result in terms of p-val)
    p=anova1(response_rate_distal_sarsa);
    p=anova1(response_rate_proximal_sarsa);
    p=anova1(response_rate_distal_fwd);
    p=anova1(response_rate_proximal_fwd);
    
end



if(0)
    
%% data save
save data_save data_behavior data_model


%% Display - behaviour profile
% 1.P(fwd)
figure('Name','P of choosing fwd')
[X,Y] = meshgrid([1:1:(length(SUB_ARRAY)*NUM_FILE_TO_PROCESS_FOR_EACH_SBJ*num_repeat_simulation+1)], [1:1:size(X_all,2)]);
half_range=max(max(max(X_all))-0.5,0.5-min(min(X_all))); % maximum dist from 0.5
STATE_mat=(ZZZ(1,:)-1)*half_range*2+0.5-half_range;
surf(X,Y,[STATE_mat' X_all'])
axis([1 size(X,2) 1 size(X,1) 0.5-half_range 0.5+half_range])
title('P(fwd choice)');
view(90,90) % view point
colormap jet
caxis([0.5-half_range 0.5+half_range])
colorbar



%% Run "mani"

% [Note] use 'X_in' for input matrix: p(m1)

% [Note] 1. Y_all(j,:)
% [PE_tolerance, Time_Step,...
% A_12, B_12, A_21, B_21,...
% param_sarsa.alpha, param_fwd.alpha, .tau(softmax),...
% myArbitrator.ind_active_model subj_ind]
% [Note] 2. Y_bestSbj: subject index that the model fits best.

mani

%% [IMPORTANT] and run this in workspace!!!!
% SIMUL_Arbitration_v3_GA_Analysis_level2(maniX,maniY,Y_all)


disp('### after mani, run     SIMUL_Arbitration_v3_GA_Analysis_level2(maniX,maniY,Y_all,X_in_fwd,X_in_sarsa,X_in_fwd_mean,X_in_fwd_var,X_in_sarsa_mean,X_in_sarsa_var)')

end




%% data collection loop (for each subject)
swsw_score=[swsw_score swsw_score_each_simul];
swsw_score_each_mode{1,swsw}=swsw_score_each_simul_each_mode;
end %swsw
%%

%% save commands
% (0) arbitration: fwd+sarsa (USE_BWDupdate_of_FWDmodel=0; USE_FWDSARSA_ONLY=0;)
% save swsw_score_NegLogLik_without_Bwdupdate swsw_score swsw_score_each_mode
% (1) arbitration: fwdbwd+sarsa (USE_BWDupdate_of_FWDmodel=1; USE_FWDSARSA_ONLY=0;)
% save swsw_score_NegLogLik_with_Bwdupdate swsw_score swsw_score_each_mode
% (2) sarsa (USE_BWDupdate_of_FWDmodel=0; USE_FWDSARSA_ONLY=2;)
% save swsw_score_NegLogLik_sarsa_only swsw_score swsw_score_each_mode
% (3) fwd (USE_BWDupdate_of_FWDmodel=0; USE_FWDSARSA_ONLY=1;)
% save swsw_score_NegLogLik_fwd_only swsw_score swsw_score_each_mode
% (4) fwdbwd (USE_BWDupdate_of_FWDmodel=1; USE_FWDSARSA_ONLY=1;)
save swsw_score_NegLogLik_fwdbwd_only swsw_score swsw_score_each_mode
%%









