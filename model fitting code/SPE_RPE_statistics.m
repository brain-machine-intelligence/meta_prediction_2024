tot_sbj=25;
real_indx=sort([2:3:105,3:3:105]);
tot_SPE=zeros(tot_sbj-1,5,3,length(real_indx)); % Total number of subjects / session number / block number is session / block profile
tot_RPE=zeros(tot_sbj-1,5,3,length(real_indx));
tot_ucs=zeros(tot_sbj-1,5,3,length(real_indx));
tot_con=zeros(tot_sbj-1,5,3);
tot_diff=zeros(tot_sbj-1,5,3);
tot_fs=zeros(tot_sbj-1,5,3);
% if exist('SBJ','var')==0
load(['\\143.248.30.101\sjh/2021winter/Behavior_Simul/task_2020/SBJ_structure_sjh_con2.mat'])
od_list=[1:7,9:25];
SBJ = SBJ2;
SBJ = SBJ(od_list);
tot_sbj = 1:length(SBJ);
indx=0;

for sbj_indx=1:tot_sbj%[1:13,15:tot_sbj]
    indx=indx+1;
    SPE_map=zeros(5,3,105); % Session number / block number in seesion  / block profile
    RPE_map=zeros(5,3,105);
    ucs_map=zeros(5,3,105);
    con_map=zeros(5,3);
    block_indx=0;
    tot_sess=length(SBJ{1,sbj_indx}.HIST_block_condition);
    for ii=1:tot_sess
        for jj=1:3
            SPE_map(ii,jj,:)=SBJ{1,sbj_indx}.regressor{1,1}.value(7,block_indx*105+1:block_indx*105+105);
            RPE_map(ii,jj,:)=SBJ{1,sbj_indx}.regressor{1,2}.value(7,block_indx*105+1:block_indx*105+105);
            for ucs_indx=1:105
                ucs_map(ii,jj,ucs_indx)=SBJ{1,sbj_indx}.HIST_block_condition{1,ii}(5,ceil(ucs_indx/3)+jj*35-35);
            end
            con_map(ii,jj)=SBJ{1,sbj_indx}.HIST_block_condition{1,ii}(3,jj*35-34);
            tot_diff(sbj_indx,ii,jj)=SBJ{1,sbj_indx}.HIST_behavior_info{1,ii}(jj*35,22);
            tot_fs(sbj_indx,ii,jj)=SBJ{1,sbj_indx}.HIST_block_condition{1,ii}(4,jj*35);
            block_indx=block_indx+1;
            tot_SPE(indx,ii,jj,:)=SPE_map(ii,jj,real_indx);
            tot_RPE(indx,ii,jj,:)=RPE_map(ii,jj,real_indx);
            tot_con(indx,ii,jj,:)=con_map(ii,jj,:);
            tot_ucs(indx,ii,jj,:)=ucs_map(ii,jj,real_indx);
        end
    end
%     tot_SPE(indx,:,:,:)=SPE_map;
%     tot_RPE(indx,:,:,:)=RPE_map;
%     tot_con(indx,:,:,:)=con_map;
    SPE_mean=zeros(tot_sess,105);
    RPE_mean=zeros(tot_sess,105);
    SPE_SD=zeros(tot_sess,105);
    RPE_SD=zeros(tot_sess,105);
    SPE_SD_ex=zeros(tot_sess,1);
    RPE_SD_ex=zeros(tot_sess,1);
    SPE_mean_ex=zeros(5,1);
    RPE_mean_ex=zeros(5,1);
    SPE_tot=[];
    RPE_tot=[];
    for kk=1:5
        SPE_temp=[];
        RPE_temp=[];
        for ii=1:4
            for jj=1:3
                if con_map(ii,jj)==kk
%                     plot(squeeze(SPE_map(ii,jj,:)))
                    SPE_temp=[SPE_temp; squeeze(SPE_map(ii,jj,:))'];
                    RPE_temp=[RPE_temp; abs(squeeze(RPE_map(ii,jj,:)))'];
                end
            end
        end
        if length(find(con_map==kk))==1
            SPE_SD(kk,:)=0; SPE_mean(kk,:)=SPE_temp;
            RPE_SD(kk,:)=0; RPE_mean(kk,:)=RPE_temp;
        else
            SPE_SD(kk,:)=std(SPE_temp); SPE_mean(kk,:)=mean(SPE_temp);
            RPE_SD(kk,:)=std(RPE_temp); RPE_mean(kk,:)=mean(RPE_temp);
        end
        SPE_SD_ex(kk)=std(nonzeros(SPE_temp)); RPE_SD_ex(kk)=std(nonzeros(RPE_temp));
        SPE_mean_ex(kk)=mean(nonzeros(SPE_temp)); RPE_mean_ex(kk)=mean(nonzeros(RPE_temp));
        SPE_tot=[SPE_tot; nonzeros(SPE_temp)]; RPE_tot=[RPE_tot; nonzeros(RPE_temp)];
    end
end
%scenario_type={'min-spe','max-spe','min-rpe','max-rpe','min-rpe-min-spe','max-rpe-max-spe','max-rpe-min-spe','min-rpe-max-spe'};
% scenarios={'SPE Min','SPE Max','RPE Min','RPE Max', 'Random'};
% real_indx=sort([2:3:105,3:3:105]);
% for ii=1:4
%     figure()
%     hold on
%     errorbar(1:length(real_indx),SPE_mean(ii,real_indx),SPE_SD(ii,real_indx))
%     errorbar(1:length(real_indx),RPE_mean(ii,real_indx)/40,RPE_SD(ii,real_indx)/40)
%     legend('SPE','RPE')
%     xlabel('Trials')
%     ylabel('Normalized')
%     ylim([-0.5 1.5])
%     title(scenarios{ii})
% end

%% Show RPE/SPE bar for single subject
% figure()
% hold on;
% errorbar(1:length(real_indx),SPE_mean(1,real_indx),SPE_SD(1,real_indx))
% errorbar(1:length(real_indx),SPE_mean(2,real_indx),SPE_SD(2,real_indx))
% legend('SPE max','SPE min')
% xlabel('Trials')
% ylabel('SPE')
% 
% figure()
% hold on;
% errorbar(1:length(real_indx),RPE_mean(3,real_indx),RPE_SD(3,real_indx))
% errorbar(1:length(real_indx),RPE_mean(4,real_indx),RPE_SD(4,real_indx))
% legend('RPE max','RPE min')
% xlabel('Trials')
% ylabel('RPE')
% 
% figure()
% hold on;
% bar(1:5, SPE_mean_ex)
% errorbar(1:5, SPE_mean_ex, SPE_SD_ex/sqrt(70), 'o')
% title('SPE')
% 
% figure()
% hold on;
% bar(1:5, RPE_mean_ex)
% errorbar(1:5, RPE_mean_ex, RPE_SD_ex/sqrt(70), 'o')
% title('RPE')
%% Total subjects SPE/RPE bar
tot_SPE_max=[]; tot_SPE_min=[]; tot_SPE_rnd=[]; tot_SPE_etc=[];
tot_RPE_max=[]; tot_RPE_min=[]; tot_RPE_rnd=[]; tot_RPE_etc=[];

tot_SPE=tot_SPE;
tot_RPE=tot_RPE;
tot_con=tot_con;
figure()
for ii=1:tot_sbj-1
    sbj_SPE_max=[]; sbj_SPE_min=[]; sbj_SPE_rnd=[]; sbj_SPE_etc=[];
    sbj_RPE_max=[]; sbj_RPE_min=[]; sbj_RPE_rnd=[]; sbj_RPE_etc=[];
    for jj=1:5
        for kk=1:3
%             tot_con(ii,jj,kk)
            switch tot_con(ii,jj,kk)
                case 1
                    tot_SPE_min=[tot_SPE_min, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_SPE_min=[sbj_SPE_min, squeeze(tot_SPE(ii,jj,kk,:))'];
                    tot_RPE_etc=[tot_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_RPE_etc=[sbj_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];
                case 2
                    tot_SPE_max=[tot_SPE_max, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_SPE_max=[sbj_SPE_max, squeeze(tot_SPE(ii,jj,kk,:))'];
                    tot_RPE_etc=[tot_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_RPE_etc=[sbj_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];                    
                case 3
                    tot_RPE_min=[tot_RPE_min, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_RPE_min=[sbj_RPE_min, squeeze(tot_RPE(ii,jj,kk,:))'];
                    tot_SPE_etc=[tot_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_SPE_etc=[sbj_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                case 4
                    tot_RPE_max=[tot_RPE_max, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_RPE_max=[sbj_RPE_max, squeeze(tot_RPE(ii,jj,kk,:))'];
                    tot_SPE_etc=[tot_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_SPE_etc=[sbj_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                case 5
                    tot_SPE_rnd=[tot_SPE_rnd, squeeze(tot_SPE(ii,jj,kk,:))'];
                    tot_RPE_rnd=[tot_RPE_rnd, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_SPE_rnd=[sbj_SPE_rnd, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_RPE_rnd=[sbj_RPE_rnd, squeeze(tot_RPE(ii,jj,kk,:))'];
                    tot_SPE_etc=[tot_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                    sbj_SPE_etc=[sbj_SPE_etc, squeeze(tot_SPE(ii,jj,kk,:))'];
                    tot_RPE_etc=[tot_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];
                    sbj_RPE_etc=[sbj_RPE_etc, squeeze(tot_RPE(ii,jj,kk,:))'];
            end
        end
    end
    sbj_RPE_min=abs(sbj_RPE_min); sbj_RPE_max=abs(sbj_RPE_max); sbj_RPE_rnd=abs(sbj_RPE_rnd); sbj_RPE_etc=abs(sbj_RPE_etc);
    subplot(5,5,ii)
    title(num2str(ii))
    hold on
    bar([mean(sbj_SPE_min),mean(sbj_SPE_max),mean(sbj_SPE_rnd)])
    errorbar(1:3,[mean(sbj_SPE_min),mean(sbj_SPE_max),mean(sbj_SPE_rnd)],[std(sbj_SPE_min)/sqrt(length(sbj_RPE_min)),std(sbj_SPE_max)/sqrt(length(sbj_RPE_max)),std(sbj_SPE_rnd)/sqrt(length(sbj_RPE_rnd))],'.')
%     bar([mean(sbj_RPE_min),mean(sbj_RPE_max),mean(sbj_RPE_rnd)])
end
% tot_RPE_min=abs(tot_RPE_min); tot_RPE_max=abs(tot_RPE_max); tot_RPE_rnd=abs(tot_RPE_rnd); tot_RPE_etc=abs(tot_RPE_etc);
figure()
subplot(1,2,1)
hold on;
bar(1:3, [mean(tot_SPE_min),mean(tot_SPE_max),mean(tot_SPE_etc)])
errorbar(1:3, [mean(tot_SPE_min),mean(tot_SPE_max),mean(tot_SPE_etc)],[std(tot_SPE_min)/sqrt(length(tot_SPE_min)),std(tot_SPE_max)/sqrt(length(tot_SPE_max)),std(tot_SPE_etc)/sqrt(length(tot_SPE_etc))],'o')
xlabel('min SPE/max SPE/others')
title('SPE')
subplot(1,2,2)
hold on;
bar(1:3, [mean(tot_RPE_min),mean(tot_RPE_max),mean(tot_RPE_etc)])
errorbar(1:3, [mean(tot_RPE_min),mean(tot_RPE_max),mean(tot_RPE_etc)], [std(tot_RPE_min)/sqrt(length(tot_RPE_max)),std(tot_RPE_max)/sqrt(length(tot_RPE_min)),std(tot_RPE_etc)/sqrt(length(tot_RPE_etc))],'o')
xlabel('min RPE/max RPE/others')
title('RPE')


%% Do the effect size measure
% addpath('C:\Users\User\Documents\MATLAB')
% con_order=reshape(con_map',1,[]);
% SPE_max_tot=SPE_tot(con_order==1,:);    SPE_max_tot=reshape(SPE_max_tot',1,[]);
% SPE_min_tot=SPE_tot(con_order==2,:);    SPE_min_tot=reshape(SPE_min_tot',1,[]);
% SPE_no_tot=SPE_tot(con_order==3 | con_order==4 ,:);    SPE_no_tot=reshape(SPE_no_tot',1,[]);
% 
% RPE_max_tot=RPE_tot(con_order==3,:);    RPE_max_tot=reshape(RPE_max_tot',1,[]);
% RPE_min_tot=RPE_tot(con_order==4,:);    RPE_min_tot=reshape(RPE_min_tot',1,[]);
% RPE_no_tot=RPE_tot(con_order==1 | con_order==2 ,:);    RPE_no_tot=reshape(RPE_no_tot',1,[]);
% 
% cohends_d(SPE_max_tot,SPE_min_tot)
% cohends_d(SPE_max_tot,SPE_no_tot)
% cohends_d(SPE_min_tot,SPE_no_tot)
% cohends_d(RPE_max_tot,RPE_min_tot)
% cohends_d(RPE_max_tot,RPE_no_tot)
% cohends_d(RPE_min_tot,RPE_no_tot)

%% Time Profile

scenarios={'SPE Min','SPE Max','RPE Min','RPE Max', 'Random'};
tot_SPE_sc=tot_SPE; tot_RPE_sc=tot_RPE; %% subject, scenario, occurence, profile
[nsbj,nsess,nblck,ntime]=size(tot_SPE);
for ii=1:nsbj
    ocurr_map=zeros(nsess,1);
    for jj=1:nsess
        for kk=1:nblck
            curr_con=tot_con(ii,jj,kk);
            if curr_con==0
                continue;
            end
            ocurr_map(curr_con)=ocurr_map(curr_con)+1;
            tot_SPE_sc(ii,curr_con,ocurr_map(curr_con),:)=tot_SPE(ii,jj,kk,:);
            tot_RPE_sc(ii,curr_con,ocurr_map(curr_con),:)=abs(tot_RPE(ii,jj,kk,:));
        end
    end
    for jj=1:nsess
        if ocurr_map(jj,1)<nblck
            tot_SPE_sc(ii,jj,ocurr_map(jj)+1:end,:)=NaN;
            tot_RPE_sc(ii,jj,ocurr_map(jj)+1:end,:)=NaN;
        end
    end
end
SPE_profile_plot=zeros(nsess,ntime,2); RPE_profile_plot=zeros(nsess,ntime,2);  %scenario, profile, mean- std
for ii=1:nsess
    for jj=1:ntime
        SPE_tmp=tot_SPE_sc(:,ii,:,jj); RPE_tmp=tot_RPE_sc(:,ii,:,jj);
        SPE_profile_plot(ii,jj,1)=mean(SPE_tmp(:),'omitnan'); SPE_profile_plot(ii,jj,2)=std(SPE_tmp(:),'omitnan')/sqrt(nsbj*nsess*nblck-1);
        RPE_profile_plot(ii,jj,1)=mean(RPE_tmp(:),'omitnan'); RPE_profile_plot(ii,jj,2)=std(RPE_tmp(:),'omitnan')/sqrt(nsbj*nsess*nblck-1);
    end
end

figure()
hold on;
for ii=1:nsess
    errorbar(1:ntime,squeeze(SPE_profile_plot(ii,:,1)),squeeze(SPE_profile_plot(ii,:,2)));
end
legend(scenarios)
title('SPE Profile')
xlabel('time points')
ylabel('SPE')
hold off

figure()
hold on;
for ii=1:nsess
    errorbar(1:ntime,squeeze(RPE_profile_plot(ii,:,1)),squeeze(RPE_profile_plot(ii,:,2)));
end
legend(scenarios)
title('RPE Profile')
xlabel('time points')
ylabel('RPE')
hold off

figure()
subplot(2,2,1)
hold on;
for ii=1:2
plot(squeeze(SPE_profile_plot(ii,:,1)));
end
title('SPE Profile')
xlabel('time points')
ylabel('SPE')
legend(scenarios(1:2))
hold off
subplot(2,2,2)
hold on;
for ii=3:4
plot(squeeze(SPE_profile_plot(ii,:,1)));
end
title('SPE Profile')
xlabel('time points')
ylabel('SPE')
legend(scenarios(3:4))
hold off
subplot(2,2,3)
hold on;
for ii=1:2
plot(squeeze(RPE_profile_plot(ii,:,1)));
end
title('RPE Profile')
xlabel('time points')
ylabel('RPE')
legend(scenarios(1:2))
hold off
subplot(2,2,4)
hold on;
for ii=3:4
plot(squeeze(RPE_profile_plot(ii,:,1)));
end
title('RPE Profile')
xlabel('time points')
ylabel('RPE')
legend(scenarios(3:4))
hold off