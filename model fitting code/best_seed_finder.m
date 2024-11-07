header='\\143.248.30.94\SJH_indi\Human_Guidance\Behavior_Simul\task_2019\';
header='\\143.248.30.94\bmlsamba\SJH_indi\Human_Guidance\Behavior_Simul\task_2019\result_simul_seed_selected_back_up\';
% od_list = [4,6:8,10,11,13,15:26];
od_list = 1:25;
file_suffix = '';
% % header = '\\143.248.30.94\bmlsamba\SJH_indi\fmri_arbitration\modelRLsource\';
% header = '\\143.248.30.101\sjh\RPE_behav\Behavior_Simul\task_2020\';
% header = '\\143.248.30.101\sjh\2021winter\Behavior_Simul\task_2020\result_simul_seed_selected\';
% header = '\\143.248.30.101\sjh\2021winter\Behavior_Simul\task_2020\';
header = '\\143.248.30.101\sjh\fmri2022\Behavior_Simul\task_2020\';
% od_list=[1:7,9:25];
file_suffix = '_extended';%'_diff_range';%'_010';

addpath(header);
max_sbj = length(od_list);
neglog=zeros(max_sbj,280);

for sbj=1:max_sbj
    for ii=1:280
        load([header 'result_simul\SBJ_structure_sbj' sprintf('%.2d',sbj) '_sjh' sprintf('%.3d',ii) file_suffix '.mat']);
        neglog(sbj,ii)=SBJ{1,1}.model_BayesArb.val;
    end
    for big_condi=1:14
        tmp=neglog(sbj,big_condi*20-19:big_condi*20);
        [min_neg_log,indx]=min(tmp);
        disp([num2str(sbj), ' ' , num2str(big_condi) ' ' num2str(min_neg_log)])
        load([header 'result_simul\SBJ_structure_sbj' sprintf('%.2d',sbj) '_sjh' sprintf('%.3d',big_condi*20-20+indx) file_suffix '.mat']);
        save([header 'result_simul_seed_selected\SBJ_structure_sbj' sprintf('%.2d',sbj) '_sjh' sprintf('_con%.2d',big_condi) file_suffix '.mat'],'SBJ');
    end
end

min_neg_log = zeros(14,max_sbj);
Num_Ocurr = zeros(14,max_sbj);
for big_condi=1:14
    SBJ2 = {};
    for sbj=od_list
%         load([header 'SBJ_structure_sbj' sprintf('%.2d',sbj) '_sjh' sprintf('_con%.2d',big_condi) file_suffix '.mat'],'SBJ');
        load([header 'result_simul_seed_selected/SBJ_structure_sbj' sprintf('%.2d',sbj) '_sjh' sprintf('_con%.2d',big_condi) file_suffix '.mat'],'SBJ');
        min_neg_log(big_condi,sbj) = SBJ{1,1}.model_BayesArb.val;
%         Num_Ocurr(big_condi,sbj) = 2*105*length(SBJ{1,1}.HIST_behavior_info);        
        Num_Ocurr(big_condi,sbj) = 2*80*length(SBJ{1,1}.HIST_behavior_info);        
        SBJ2(1,sbj) = SBJ;
    end
    save([header 'SBJ_structure_sjh_con' num2str(big_condi) file_suffix '.mat'], 'SBJ2')
end


%max_sbj = 26;
BICs=zeros(14,max_sbj);
num_params=[2,2,6,4,4,8,6,6,6,4,4,8,6,6];
for con=1:14
    for ii= od_list
        for sc=1:5
            BICs(con,ii)=log(Num_Ocurr(con,ii))*num_params(con)+2*min_neg_log(con,ii);
        end
    end
end
save([header 'BICs'  file_suffix  '.mat'],'BICs');
BICs_median=zeros(14,1);
BICs_sem=zeros(14,1);

for con=1:14
    BICs_median(con)=median(squeeze(BICs(con,:)));
    BICs_sem(con)=std(squeeze(BICs(con,:)))/sqrt(max_sbj);
end

figure()
hold on
bar(BICs_median)
errorbar(1:14,BICs_median,BICs_sem,'.')
ylim([min(BICs_median-BICs_sem)-10 max(BICs_median+BICs_sem)+10])
hold off;

params = [];

load([header 'SBJ_structure_sjh_con9' file_suffix '.mat'])
for ii = 1:length(od_list)
    params = [params; SBJ2{1,ii}.model_BayesArb.param];    
end

LIST_titles = {'0-PE','|RPE| lr','AMB2MF', 'AMF2MB', 'inverseT', 'lr'};
figure()
for ii = 1:6
    subplot(2,3,ii)
    histogram(params(:,ii))
    title(LIST_titles{ii})
end