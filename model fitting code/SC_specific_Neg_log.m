max_sbj=26;
SC_Neg_Log=zeros(14,max_sbj,5);
SC_Num_Ocurr=zeros(14,max_sbj,5);
for con=1:14
    disp(con)
    file_name=['\\143.248.30.94\bmlsamba\SJH_indi\Human_Guidance\Behavior_Simul\task_2019\result_simul_seed_selected_back_up\total_SBJ_con' sprintf('%.2d', con) '.mat'];
    load(file_name);
    for ind=1:max_sbj
        SBJ_indx{1,1}=SBJ{1,ind};
        mode=SBJ_indx{1,1}.model_BayesArb.mode;
        mode.out=2;
        eval_out=eval_ArbitrationRL3(SBJ_indx{1,1}.model_BayesArb.param, SBJ_indx, mode);
        SC_Neg_Log(con,ind,:)=squeeze(eval_out(1,:));
        SC_Num_Ocurr(con,ind,:)=squeeze(eval_out(2,:));
    end
end

SC_NL_median=zeros(14,5);
SC_NL_sem=zeros(14,5);

for con=1:14
    for sc=1:5
        SC_NL_median(con,sc)=median(squeeze(SC_Neg_Log(con,:,sc)));
        SC_NL_sem(con,sc)=std(squeeze(SC_Neg_Log(con,:,sc)))/sqrt(max_sbj);
    end
end
scenarios={'min SPE', 'max SPE' ,'min RPE', 'max RPE', 'random'};
for ii=1:5
    figure()
    hold on;
    bar(squeeze(SC_NL_median(:,ii)));
    errorbar(1:length(SC_NL_median(:,ii)),squeeze(SC_NL_median(:,ii)),squeeze(SC_NL_sem(:,ii)),'');
    title(scenarios{1,ii})
end

SC_BICs=zeros(14,max_sbj,5);
num_params=[2,2,6,4,4,8,6,6,6,4,4,8,6,6];
for con=1:14
    for ii= 1:max_sbj%26:27
        for sc=1:5
            SC_BICs(con,ii,sc)=log(SC_Num_Ocurr(con,ii,sc)/2)*num_params(con)+2*SC_Neg_Log(con,ii,sc);
        end
    end
end


SC_BICs_median=zeros(14,5);
SC_BICs_sem=zeros(14,5);

sc=1;
for con=1:14
    for sc=1:5
        SC_BICs_median(con,sc)=median(squeeze(SC_BICs(con,[1:13,15:end],sc)));
        SC_BICs_sem(con,sc)=std(squeeze(SC_BICs(con,[1:13,15:end],sc)))/sqrt(max_sbj);
    end
end
% for con=1:14
%     for sc=2:5
%         SC_BICs_median(con,sc)=median(squeeze(SC_BICs(con,:,sc)));
%         SC_BICs_sem(con,sc)=std(squeeze(SC_BICs(con,:,sc)))/sqrt(max_sbj);
%     end
% end
scenarios={'min SPE', 'max SPE' ,'min RPE', 'max RPE', 'random'};
for ii=1:5
    figure()
    hold on;
    bar(squeeze(SC_BICs_median(:,ii)));
    errorbar(1:length(SC_BICs_median(:,ii)),squeeze(SC_BICs_median(:,ii)),squeeze(SC_BICs_sem(:,ii)),'');
    title(['BIC : ' scenarios{1,ii}])
end

scenarios={'min SPE', 'max SPE' ,'min RPE', 'max RPE', 'random'};
figure()
for ii=1:4
    subplot(2,2,ii)
    hold on;
    bar(squeeze(SC_BICs_median(:,ii)));
    errorbar(1:length(SC_BICs_median(:,ii)),squeeze(SC_BICs_median(:,ii)),squeeze(SC_BICs_sem(:,ii)),'o');
    title(['BIC : ' scenarios{1,ii}])
    ylim([0 500])
end
BIC_pval=zeros(14,14,5);
for sc=1:5
    for ii=1:14
        for jj=1:14
            [~,BIC_pval(ii,jj,sc)]=ttest(SC_BICs(ii,:,sc),SC_BICs(jj,:,sc));
        end
    end
end
BIC_rank=zeros(14,5);
for ii=1:5
    [~,BIC_rank(:,ii)]=sort(SC_BICs_median(:,ii));
end