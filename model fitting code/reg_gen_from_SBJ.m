is_on_cluster= false;
SBJ = SBJ2;
SBJ_loaded = 1; % 0: no SBJ file so need to create from behavior, 1: We loaded SBJ and just need to extract the regressors
list_sbj_included=1:27; %% Currently Regressors works for ~21, and needed to fix to ~26
if is_on_cluster
    header='/home';
else
    header='\\143.248.30.101\sjh';
end
addpath([header, '/spm12']);
con = 1;
tag = '_con01'
% plug-in
warning('off')
%list_sbj_included=job_opt.list_sbj_included;
%
% path1=[header '/SJH_indi']; % Use for 2014 & 2019
% path1=[header '/syh1743']; % use for dep
% path1=[header '/RPE_behav']; % use for dep
path1=[header '/2021winter']; % use for dep
% path0=[header, '/SJH_indi/Human_Guidance/Behavior_Simul']; % Use for 2019
% path0=[header, '/SJH_indi/fmri_arbitration/']; % Use for 2014
% path0=[header, '/SJH_indi/Depression_ana/']; % Use for 2014
path0=[header, '/2021winter/Behavior_Simul/task_2020']; % Use for 2014
% save_path_result : , storage for regressors
% save_path_result=[path0 '/result_simul/']; % Use for 2019/2014
% save_path_result=[path0 '/result_simul/']; % Use for dep
save_path_result=[path0, '/regressors_contrasts/'];
save_for_SPM=[path0, '/regressors_contrasts/'];
save_path_neuroecon=[path0 '/regressors_contrasts/'];
% is_on_cluster=job_opt.is_on_cluster;
mode.path_ext=path0;
addpath(path1);
addpath(path0);    

% 1. Behavioral data
% LIST_SBJ={'david', 'DeDe', 'rosemary', 'Boyu', 'melissa', 'Rehevolew', 'joel', 'clarke', 'angela', 'william', 'josephine'}; % (good in pre which is mostly habitual - rosemary, melissa)
% mode.map_type=?;

% 2. behavioral + fmri data (for map config, see SIMUL_arbitraion_fmri2.m)
% [note] 'Oliver' uses an old map. the rest of them use a new map.
% LIST_SBJ={'CYJ'};
% LIST_SBJ={'Oliver', 'Hao', 'Breanna', 'Derek', 'Timothy', 'Teagan', 'Jeffrey', 'Seung', 'Carole', 'Tony', 'Surendra', 'Lark',...                                % Use for 2014
%     'Joaquin', 'DavidB', 'Christopher', 'Gjergji', 'Charles', 'Erin', 'Connor', 'Domenick', 'Thao', 'Arin', 'Pauline', 'Tho','CYJ','pilotP','pilotM','pilotS','HLJ'}; % Use for 2014
% LIST_SBJ={'sbj1','sbj3','sbj11','sbj12','sbj18','sbj19','sbj20','sbj24','sbj25','sbj27','sbj32','sbj33','sbj34','sbj35',...                       % Use for Dep
%     'sbj37','sbj38','sbj40','sbj41','sbj42','sbj44','sbj46','sbj48_kdy','sbj50_kdh','sbj51_akr','sbj52_jkw','sbj53_nyk','sbj54_jyh','sbj55_ker'}; % Use for Dep
%LIST_SBJ_pilot={'CYJ','pilotP','pilotM'};
% LIST_SBJ={'sbj001_SYH','sbj002_OJH','sbj003_HJH','sbj004_KMS','sbj005_LSB','sbj006_JJM','sbj007_KMJ','sbj008_KHY','sbj009_WJI','sbj010_HSJ','sbj011_PSW','sbj012_KKY','sbj013_KYL','sbj014_HSL','sbj015_MHJ','sbj016_YSH','sbj017_LHK', 'sbj018_JHH','sbj019_KHJ'...
%     ,'sbj020_IYS','sbj021_JSM','sbj022_KHK','sbj023_KJS','sbj024_JHC',};
% LIST_SBJ = {'subject001_ksy','subject03_keb','subject004_ksj','subject5_lwc','subject6_sjh','subject7_kch','subject8_kjs','subject9_lks','subject10_ssh','subject11_hjy','subject12_khs','subject13_pjb','subject14_syh','subject15_kik','subject16_jja','subject16_lsh','subj18_kjh','subj19_kny','subj20_jjh','subject21_lsl','subject22_jsh','subject23_yhj','subj24_kjy','subj25_kjs','subject26_cjb','subj27_kkm','subject28_ljh','subject29_cyj','subject30','subj31_ssw','subj32_khj','subj33_ohy','subj34_yhw','subj35_ajs','subj37_shi','subj38_lsh','subj40_kkr','subj41_jhj', 'sbj42_ljk','sbj44_ksy', 'sbj45_jej','sbj46_ses','sbj47_hej','sbj48_kdy','subject49_ljy','sbj50_kdh','sbj51_akr', 'sbj52_jkw','sbj53_nyk','sbj54_jyh', 'sbj55_ker','sbj56_pjw','sbj57_kjs','sbj58_jye','sbj59_hst','sbj60_kdh','sbj61_kis','sbj62_kms','sbj63_njy','sbj64_lbj','sbj65_syh','sbj66_ljh','sbj67_ljs'};
% LIST_SBJ = {'subject001_ksy','subject03_keb','subject004_ksj','subject5_lwc','subject6_sjh','subject7_kch','subject8_kjs','subject9_lks','subject10_ssh','subject11_hjy','subject12_khs','subject13_pjb','subject14_syh','subject15_kik','subject16_jja','subject16_lsh','subj18_kjh','subj19_kny','subj20_jjh','subject21_lsl','subject22_jsh','subject23_yhj','subj24_kjy','subj25_kjs','subject26_cjb','subj27_kkm','subject28_ljh','subject29_cyj','subject30','subj31_ssw','subj32_khj','subj33_ohy','subj34_yhw','subj35_ajs','subj37_shi','subj38_lsh','subj40_kkr','subj41_jhj', 'sbj42_ljk','sbj44_ksy', 'sbj45_jej','sbj46_ses','sbj47_hej','sbj48_kdy','subject49_ljy','sbj50_kdh','sbj51_akr', 'sbj52_jkw','sbj53_nyk','sbj54_jyh', 'sbj55_ker','sbj56_pjw'};
% LIST_SBJ={'202108021500','202108051045','202108051230','202108090900','202108091415','202108091600','202108120900','202108121415'};
LIST_SBJ={'KSM','CJS','KCY','KMS','SHA','KAY','SHC','KDE','SYB','CYJ','KYY','YCS','PYJ','KCY2',...
            'LZY','OHJ','PYJ2','MYL','LGH','LHN','LSL','LHM','SJY','YSY','CHK','MCH','KDH'};   
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
%% OPTION - Regressor arrangement
% {'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd', 'Qarb','uncertaintyM2', 'invFanoM2', 'duncertaintyM1', 'dinvFanoM1', 'weigtM1'};
% should add the regressors in the order of importance
% param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'invFanoM1', 'Qsarsa','Qfwd','Qarb','uncertaintyM2', 'invFanoM2', 'weigtM1'};
% param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'MaxinvFano12','dinvFano12','Qsarsa', 'Qfwd', 'Qarb', 'dQbwdEnergy', 'weigtM1', 'invF1_corrected', 'invF2_corrected'};
switch con
    case 1
        param_regressor_type_cue={'SPE', 'RPE', 'Qsarsa'};
    case 2
        param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'MaxinvFano12','dinvFano12','Qsarsa', 'Qfwd', 'Qarb', 'dQbwdEnergy', 'invF1_corrected', 'invF2_corrected'};
    otherwise
        param_regressor_type_cue={'SPE', 'RPE', 'uncertaintyM1', 'uncertaintyM2', 'MaxinvFano12','dinvFano12','Qsarsa', 'Qfwd', 'Qarb', 'dQbwdEnergy', 'weigtM1', 'invF1_corrected', 'invF2_corrected'};
end
disp(param_regressor_type_cue)
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
if SBJ_loaded == 0 %There is no pre-existing SBJ so need to create it from behavioral data
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
        SBJ{1,i}.HIST_block_condition_pre=HIST_block_condition;
        SBJ{1,i}.HIST_behavior_info_pre=HIST_behavior_info;
        SBJ{1,i}.HIST_map_state_info_pre=HIST_map_state_info;


        % 'fmri' file load: HIST_block_condition{1,session_ind}, HIST_behavior_info{1,session_ind}
        file_name=[LIST_SBJ{1,ind_sbj_included(i)} '_fmri_info.mat'];
        if(is_on_cluster==0)
            file_name_full=[mode.path_ext '\result_save\' file_name];
        else
            file_name_full=[mode.path_ext '/result_save/' file_name];
        end
        load(file_name_full);
        SBJ{1,i}.HIST_behavior_info=HIST_behavior_info;
        SBJ{1,i}.HIST_behavior_info_Tag=HIST_behavior_info_Tag;
        SBJ{1,i}.HIST_event_info=HIST_event_info;
        SBJ{1,i}.HIST_event_info_Tag=HIST_event_info_Tag;
        SBJ{1,i}.HIST_block_condition=HIST_block_condition;
        SBJ{1,i}.HIST_block_condition_Tag=HIST_block_condition_Tag;
        SBJ{1,i}.HIST_map_state_info=HIST_map_state_info;
        SBJ{1,i}.HIST_map_state_info_pre_Tag=HIST_map_state_info_Tag;
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
end
    

    SBJ=SBJ(1,list_sbj_included);
    for jj2=1:1:size(SBJ,2)        % each subject
        disp(sprintf('##### creating regressor structures (sbj%02d/%02d) #######',jj2,size(SBJ,2)));
        for kk2=1:1:size(SBJ{1,jj2}.HIST_behavior_info,2)  % each main session
            
            
            
            mat_work=SBJ{1,jj2}.HIST_event_info{1,kk2};
            num_tot_events=size(SBJ{1,jj2}.HIST_behavior_info{1,kk2},1);
            
            
            
            %% 1. Regressor cue presentation - with parametric modulation (timing: stimulus onset)
            % [0-duration] SPE, RPE
            % [RT-duration] Q_fwd, Q_sarsa
            
            % regressor generation for each main session and save it to a single file that is compatible with SPM
            ind_reg=0;  % corresponds to the size of the structure
            ind_reg_abs=0; % actual number of regressors (including parametric)
            durations={};
            onsets={};
            names={};
            pmod=struct('name',{},'param',{},'poly',{});
            
            
            use_model_regressor_cue=1;
            
            % (1) durations, name, onset
            [tmp col_on]=find((mat_work(7,:)==1)|(mat_work(7,:)==2)|(mat_work(7,:)==3)|(mat_work(7,:)==4)|(mat_work(7,:)==5)...
                |(mat_work(7,:)==5.9)|(mat_work(7,:)==6.1)|(mat_work(7,:)==6.9)|(mat_work(7,:)==7.1)|...
                (mat_work(7,:)==7.9)|(mat_work(7,:)==8.1)|(mat_work(7,:)==9));
            if(length(col_on)~=num_tot_events*3)
                error('-ERROR: variable ''mat_work'' missed some event extraction. check!')
            end
            
            RT_mat=[];  onset_mat=[];
            prev_trial=0;  show_n_th_times_t=0;
            param_mat{1,1}=zeros(length(ind_regressor_type_base{1,1}.ind_reg),length(col_on));
            param_mat{1,2}=zeros(length(ind_regressor_type_base{1,2}.ind_reg),length(col_on));
            
            for ll2=1:1:length(col_on)
                
                if(ll2<length(col_on)) % usual case
                    pt_on=mat_work(4,col_on(ll2));
                    col_off=col_on(ll2)+1;
                    %                 col_off=col_on(ll2)-1+find(mat_work(7,[col_on(ll2):1:(col_on(ll2)+2)])==0.5); % find the next fixation mark presentation
                    pt_off=mat_work(4,col_off);
                    RT=pt_off-pt_on;
                else % last event in the session is the outcome presentation
                    RT=2.0;
                end
                RT_mat=[RT_mat RT];
                onset_t=mat_work(4,col_on(ll2));
                onset_mat=[onset_mat onset_t];
                
                % fill out regresssor values
                if(use_model_regressor_cue==1)
                    
                    % regressor type1
                    for nn=1:1:length(ind_regressor_type_base{1,1}.ind_reg)
                        mysession=kk2;
                        myblock=mat_work(1,col_on(ll2)); % block in session
                        mytrial=mat_work(2,col_on(ll2)); % trial in block
                        mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial (arbitration index: 1 at the second stage, 2 at the third stage)
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        param_mat{1,1}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,1}.ind_reg(nn)),col_event);
                        
                    end
                    
                    
                    % regressor type2
                    for nn=1:1:length(ind_regressor_type_base{1,2}.ind_reg)
                        mysession=kk2;
                        myblock=mat_work(1,col_on(ll2)); % block in session
                        mytrial=mat_work(2,col_on(ll2)); % trial in block
                        mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        param_mat{1,2}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,2}.ind_reg(nn)),col_event);
                    end
                    
                end
                
            end
            
            
            % 1st regressors
            ind_reg=ind_reg+1;
            onsets{1,ind_reg}=onset_mat;
            if(reg_type_go_first(1)==1)            names{1,ind_reg}=['Cue_{0T}'];  durations{1,ind_reg}=0;        end
            if(reg_type_go_first(1)==1.5)        names{1,ind_reg}=['Cue_{1T}'];    durations{1,ind_reg}=RT_mat;       end
            
            ind_reg_abs=1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
            
            
            % (2) pmod: how many times each cue presented
            if(use_model_regressor_cue==1)
                for nn=1:1:length(ind_regressor_type_base{1,1}.ind_reg)
                    pmod(1,ind_reg).name{1,nn}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,1}.ind_reg(nn)}.name];
                    pmod(1,ind_reg).poly{1,nn}=1;
                    pmod(1,ind_reg).param{1,nn}=abs(param_mat{1,1}(nn,:));
                    ind_reg_abs=ind_regressor_type_base{1,1}.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,nn};
                end
            end
            
            
            % 2nd regressors
            ind_reg=ind_reg+1;
            onsets{1,ind_reg}=onset_mat;
            if(reg_type_go_first(2)==1)            names{1,ind_reg}=['Cue_{0T}'];  durations{1,ind_reg}=0;        end
            if(reg_type_go_first(2)==1.5)        names{1,ind_reg}=['Cue_{1T}'];    durations{1,ind_reg}=RT_mat;       end
            ind_reg_abs=ind_regressor_type_base{1,1}.abs_pos_in_design_mat(end)+1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
            
            % (2) pmod: 
            if(use_model_regressor_cue==1)
                ind_reg_param=0;
                for nn=1:1:length(ind_regressor_type_base{1,2}.ind_reg)
                    ind_reg_param=ind_reg_param+1;
                    pmod(1,ind_reg).name{1,ind_reg_param}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,2}.ind_reg(ind_reg_param)}.name];
                    pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                    pmod(1,ind_reg).param{1,ind_reg_param}=abs(param_mat{1,2}(ind_reg_param,:));
                    ind_reg_abs=ind_regressor_type_base{1,2}.abs_pos_in_design_mat(ind_reg_param);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                end
                % Reserve spaces for extra regressors (will come from the dummy design matrix)
                if(reg_type_go_first(3)==2)
                    for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=ind_regressor_type_dummy.name{1,nn};
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=rand(1,length(param_mat{1,1}(1,:)));
                        %                         ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                end
            end
            
            
            onset_event=onsets;
            
            
            %% 1-A. Regressor cue presentation - regressor type 3 only!!! with parametric modulation (timing: decision time)            
            % [0-duration] 
            % [RT-duration]
            
            if(reg_type_go_first(3)==1.7)
                
                use_model_regressor_cue=1;
                
                % (1) durations, name, onset
                [tmp col_on]=find((mat_work(7,:)==10)|(mat_work(7,:)==11));
                
                RT_mat=[];  onset_mat=[];
                prev_trial=0;  show_n_th_times_t=0;
                param_mat3{1,1}=zeros(length(ind_regressor_type_base{1,3}.ind_reg),length(col_on));
                
                
                for ll2=1:1:length(col_on)
                    
                    if(ll2<length(col_on)) % usual case
                        pt_on=mat_work(4,col_on(ll2));
                        col_off=col_on(ll2)+1;
                        %                 col_off=col_on(ll2)-1+find(mat_work(7,[col_on(ll2):1:(col_on(ll2)+2)])==0.5); % find the next fixation mark presentation
                        pt_off=mat_work(4,col_off);
                        RT=pt_off-pt_on;
                    else % last event in the session is the outcome presentation
                        RT=2.0;
                    end
                    RT_mat=[RT_mat RT];
                    onset_t=mat_work(4,col_on(ll2));
                    onset_mat=[onset_mat onset_t];
                    
                    % fill out regresssor values
                    if(use_model_regressor_cue==1)
                        
                        % regressor type3 only
                        for nn=1:1:length(ind_regressor_type_base{1,3}.ind_reg)
                            mysession=kk2;
                            myblock=mat_work(1,col_on(ll2)); % block in session
                            mytrial=mat_work(2,col_on(ll2)); % trial in block
                            mytrial_s=mat_work(3,col_on(ll2)); % trial_s in trial (arbitration index: 1 at the second stage, 2 at the third stage)
                            mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(nn)}.value(1:4,:);
                            identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                            col_event=find(identity_tmp==0);
                            param_mat3{1,1}(nn,ll2)=SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(nn)}.value(row_mat(ind_regressor_type_base{1,3}.ind_reg(nn)),col_event);
                        end
                        
                    end
                    
                end
                
                
                % 3rd regressors
                ind_reg=ind_reg+1;
                onsets{1,ind_reg}=onset_mat;
                if(reg_type_go_first(3)==1.7)            names{1,ind_reg}=['Decision_{0T}'];  durations{1,ind_reg}=0;        end
                
                ind_reg_abs=ind_regressor_type_base{1,2}.abs_pos_in_design_mat(end)+1;      list_name_for_contrast{1,ind_reg_abs}=names{1,ind_reg}; % add to the global list of regresssors
                
                
                % (2) pmod:
                if(use_model_regressor_cue==1)
                    ind_reg_param=0;
                    for nn=1:1:length(ind_regressor_type_base{1,3}.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_base{1,3}.ind_reg(ind_reg_param)}.name];
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=abs(param_mat3{1,1}(ind_reg_param,:));
                        ind_reg_abs=ind_regressor_type_base{1,3}.abs_pos_in_design_mat(ind_reg_param);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                    % Reserve spaces for extra regressors (will come from the dummy design matrix)
                    for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                        ind_reg_param=ind_reg_param+1;
                        pmod(1,ind_reg).name{1,ind_reg_param}=ind_regressor_type_dummy.name{1,nn};
                        pmod(1,ind_reg).poly{1,ind_reg_param}=1;
                        pmod(1,ind_reg).param{1,ind_reg_param}=rand(1,length(param_mat3{1,1}(1,:)));
%                         ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,ind_reg_param};
                    end
                end
                
            end
            
            %%
            
            
            
            
            % (3) Saving normal regressor file
            tot_num_myregressor=length(list_name_for_contrast);
            save_file_name=['Regressor--' SBJ{1,jj2}.name '_sess' sprintf('%02d',kk2) tag '.mat'];
            if(Is_save_files_local==1)
                eval(['save ' save_path_result save_file_name ' durations names onsets pmod'])
                eval(['save ' save_for_SPM save_file_name ' durations names onsets pmod'])
            end
            if(Is_save_files_cluster==1)
                eval(['save ' save_path_neuroecon save_file_name ' durations names onsets pmod'])
            end
            
            
            
            %% 2. Extra Regressors independent of cue presentation (timing: TR)
            % The regressors will be saved in separate .mat file.
            
            % regressor generation for each main session and save it to a single file that is compatible with SPM
            ind_reg=0;  % corresponds to the size of the structure
            ind_reg_abs=0; % actual number of regressors (including parametric)
            durations={};
            onsets={};
            names={};
            pmod=struct('name',{},'param',{},'poly',{});
            
            
            ind_reg=ind_reg+1;
            TR_CBIC=2.78; % (sec)
            % (1) duration,onset,name
            durations{1,ind_reg}=TR_CBIC;
            onsets{1,ind_reg}=[0:TR_CBIC:(mat_work(4,end)+20)];
            names{1,ind_reg}=['Dummy'];
            % the name will NOT be added to "list_name_for_contrast" because it is a dummy regressor.
            length_reg=length(onsets{1,1});
            % (2-1) determine the regressor values
            regressor_extra=zeros(length(ind_regressor_type_dummy.ind_reg),length_reg);
            for nn=1:1:length(ind_regressor_type_dummy.ind_reg) % for each regressor
                % find the previous onset point closest to the current time
                t_scan=0;
                for i_scan=1:1:length_reg
                    % find the recent onset (event) time on which the regressor value has been updated.
                    test_mat=onset_event{1,1}-t_scan;
                    test_ii=find(test_mat<0);
                    if(length(test_ii)==0) % first a few scans before the event
                        myscan_time=onset_event{1,1}(1);
                    else % use the value of previous event
                        myscan_time=onset_event{1,1}(min(length(onset_event{1,1}),test_ii(end)));
                    end
                    % find the corresponding {block#, trial#, and trial_s#} in event matrix
                    mycol=find(abs(mat_work(4,:)-myscan_time)<0.001);
                    mysession=kk2; % session
                    myblock=mat_work(1,mycol); % block in session
                    mytrial=mat_work(2,mycol); % trial in block
                    mytrial_s=mat_work(3,mycol); % trial_s in trial (mytrial_s=max(1,mat_work(3,mycol)-1);)
                    % find the corresponding regressor value in a regressor matrix
                    if(mytrial_s~=1)
                        mat_work_reg=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(1:4,:);
                        identity_tmp=sum(abs(mat_work_reg-repmat([mysession myblock mytrial mytrial_s]',1,size(mat_work_reg,2))));
                        col_event=find(identity_tmp==0);
                        regressor_extra(nn,i_scan)=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(row_mat(ind_regressor_type_dummy.ind_reg(nn)),col_event);
                    else % mytrial_s=1: the first state
                        if(i_scan==1) % for the very first scan, we simply read the first regressor value.
                            regressor_extra(nn,i_scan)=SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.value(row_mat(ind_regressor_type_dummy.ind_reg(nn)),1);
                        else % for the first state in every trial, we simply take the regressor value at t-1 (because there is no update)
                            regressor_extra(nn,i_scan)=regressor_extra(nn,i_scan-1);
                        end
                    end
                    % compute the next scan time
                    t_scan=t_scan+TR_CBIC;
                end
            end
            % (2-2) pmod: parametric modulators
            for nn=1:1:length(ind_regressor_type_dummy.ind_reg)
                pmod(1,ind_reg).name{1,nn}=[SBJ{1,jj2}.regressor{1,ind_regressor_type_dummy.ind_reg(nn)}.name];
                pmod(1,ind_reg).poly{1,nn}=1;
                pmod(1,ind_reg).param{1,nn}=abs(regressor_extra(nn,:));
                ind_reg_abs=ind_regressor_type_dummy.abs_pos_in_design_mat(nn);      list_name_for_contrast{1,ind_reg_abs}=pmod(1,ind_reg).name{1,nn};
            end
            
            % (3) Saving dummy regressor file
            tot_num_myregressor=length(list_name_for_contrast);
            save_file_name=['Regressor_dummy--' SBJ{1,jj2}.name '_sess' sprintf('%02d',kk2) tag '.mat'];
            if(Is_save_files_local==1)
                eval(['save ' save_path_result save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
                eval(['save ' save_for_SPM save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
            end
            if(Is_save_files_cluster==1)
                eval(['save ' save_path_neuroecon save_file_name ' durations names onsets pmod ind_regressor_type_dummy'])
            end
            
        end
    end
    
    
    %% Saving Contrast file
    % [index of my regressors for contrast vector] : total main regressor=6, total regressors=7
    clear contrast_spm
    
    total_number_regressor=tot_num_myregressor+6; % # + 6 movements
    ind_contrast_vec=0;
    
    
    % individual : (ex) [0 1 0 0 0 0 0 0]
    for ii=1:1:tot_num_myregressor
        ind_contrast_vec=ind_contrast_vec+1;
        contrast=zeros(1,tot_num_myregressor);
        contrast(1,ii)=1;
        contrast_spm{1,ind_contrast_vec}.name=list_name_for_contrast{1,ii};
        contrast_spm{1,ind_contrast_vec}.vec=contrast;
    end
    
    % % difference : (ex) [0 0 0 0 1 -1 0 0]
    % combination_mat=combnk(param_regressor_type_cue_abs_pos_in_design_mat,2);
    % for kk=1:1:size(combination_mat,1)
    %     % A-B
    %     ind_contrast_vec=ind_contrast_vec+1;
    %     contrast=zeros(1,total_number_regressor);
    %     contrast(1,combination_mat(kk,1))=1;    contrast(1,combination_mat(kk,2))=-1;
    %     contrast_spm{1,ind_contrast_vec}.name=[list_name_for_contrast{1,combination_mat(kk,1)} '>' list_name_for_contrast{1,combination_mat(kk,2)}];
    %     contrast_spm{1,ind_contrast_vec}.vec=contrast;
    %     % B-A
    %     ind_contrast_vec=ind_contrast_vec+1;
    %     contrast=zeros(1,total_number_regressor);
    %     contrast(1,combination_mat(kk,2))=1;    contrast(1,combination_mat(kk,1))=-1;
    %     contrast_spm{1,ind_contrast_vec}.name=[list_name_for_contrast{1,combination_mat(kk,1)} '<' list_name_for_contrast{1,combination_mat(kk,2)}];
    %     contrast_spm{1,ind_contrast_vec}.vec=contrast;
    % end
    
    if(Is_save_files_local==1)
        eval(['save ' save_path_result 'contrast_spm' tag '.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
        eval(['save ' save_for_SPM 'contrast_spm' tag '.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
    end
    if(Is_save_files_cluster==1)
        eval(['save ' save_path_neuroecon 'contrast_spm' tag '.mat' ' contrast_spm ind_regressor_type_base ind_regressor_type_dummy param_regressor_type_cue_abs_pos_in_design_mat list_name_for_contrast'])
    end