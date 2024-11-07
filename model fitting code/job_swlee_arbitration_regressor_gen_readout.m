% clear all;
% close all;
%% ID, Password, folderpath
addpath('\\143.248.30.101\sjh\kdj\TerminalControl');
id = 'sjh';
pw = 'kinggodjh';
path0='\\143.248.30.101\sjh\basic_codes\model_codes';
%cd('\\143.248.30.94\bmlsamba/SJH_indi/fmri_arbitration')
addpath(path0)
cd(path0);
%% Parameters (global)
% list_subjects={'Oliver', 'Hao', 'Breanna', 'Derek', 'Timothy', 'Teagan', 'Jeffrey', 'Seung', 'Carole', 'Tony', 'Surendra', 'Lark',...
%     'Joaquin', 'DavidB', 'Christopher', 'Gjergji', 'Charles', 'Erin', 'Connor', 'Domenick', 'Thao', 'Arin', 'Pauline', 'Tho','CYJ','pilotP','pilotM'}; % should be placed in scanning order
% available_cluster_core = [1,2,4,4];
available_cluster_core = [1];
job.data_type = '2021'; %'2020'; %'2014','dep','2019', '2022'
only_40 = 0;
%% Inputs
%13-17 running->3 / 1~2 3~7->1 8~12->2 13~17->3(FURTHER)

switch job.data_type
    case '2014'
       sbj_included=1:24;
       sbj_included = 1;
        for jj=1:24
            for parameter_indx=[1:20:280]%1:280%221:240%1:280
                job.nth=jj*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_2019=0;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_2019,parameter_indx,only_40,job.data_type};
                JobCreate(id,job,1,ceil(jj/12));
                [out] = SubmitJob(id,pw,job);   
            end
        end
    case 'dep'

        sbj_included=1:28;
        sbj_included = 1;
        for jj=1:28
            for parameter_indx=1:20:280 %1:280%221:240%1:280
                job.nth=jj*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_2019=0;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_2019,parameter_indx,only_40,job.data_type};
                JobCreate(id,job,1,ceil(jj/13));
                [out] = SubmitJob(id,pw,job);   
            end
        end
        
    case '2019'

        sbj_included=1:27;%1:24;%[2:1:19 21:1:24]; %%%%%%22~26도 돌려야함 ㅅㄱ
        sbj_included = 1;
        for jj=27%1:26
            for parameter_indx=1:20:280 %1:280%221:240%1:280
                job.nth=jj*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_type=2019;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_type,parameter_indx,only_40,job.data_type};
                JobCreate(id,job,1,ceil(jj/13));
                [out] = SubmitJob(id,pw,job);   
            end
        end
    case '2020'

        sbj_included=1:8;%1:24;%[2:1:19 21:1:24]; %%%%%%22~26도 돌려야함 ㅅㄱ
        sbj_included = 1;
        for jj=1:8
            for parameter_indx=1:20:280 %1:280%221:240%1:280
                job.nth=jj*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_type=2020;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_type,parameter_indx,only_40,job.data_type};
                JobCreate(id,job,1,rem(jj,2)+1);
                [out] = SubmitJob(id,pw,job);   
            end
        end
        
        case '2021'

        sbj_included=1:27; %1~17
        sbj_included = 1;
        for jj=1:length(sbj_included)
            for parameter_indx=1:20:280 %1:280%221:240%1:280
                job.nth=sbj_included(jj)*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_type=2020;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_type,parameter_indx,only_40,job.data_type};
                JobCreate(id,job,1);%,rem(jj,4)+1);
                [out] = SubmitJob(id,pw,job);   
            end
        end
        case '2022'

        sbj_included=1:25; %1~17
        sbj_included = 1;
        for jj=1:length(sbj_included)
            for parameter_indx=1:20:280 %1:280%221:240%1:280
                job.nth=sbj_included(jj)*1000+parameter_indx;
                job.name = 'batch_model_arbitration_regressor_gen_v6_sjh_readout';
                job.path0 = '/home/sjh/basic_codes/model_codes';
                chara='server';
                if strcmp(chara,'server')
                    job_opt.is_on_cluster=1;
                elseif strcmp(chara,'local')
                    job_opt.is_on_cluster=0;
                end
                job_opt.task_type=2020;
                job_opt.list_sbj_included=sbj_included(jj);
                job.argu = {job_opt.list_sbj_included,job_opt.is_on_cluster,job_opt.task_type,parameter_indx,only_40,job.data_type};
                if rem(jj,2) == 1
                    JobCreate(id,job,1,1);
                else
                    JobCreate(id,job,1,2);
                end
                [out] = SubmitJob(id,pw,job);   
            end
        end
end

disp('### All jobs have been submitted. Use "job_info(t_all)" for more details  or "job_del([id])" for deleting log files. ###');
