function SIMUL_cmd_send(simul_set,TASK_TYPE,ii)

    addpath('\\143.248.30.101\sjh\RPE_pols\TerminalControl');

    id = 'sjh';
    pw = 'kinggodjh';
    control_mode = {'max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe','max-MB','min-MB','max-MF','min-MF','min-MF-min-MB','max-MF-max-MB','min-MF-max-MB','max-MF-min-MB'};
%     if simul_set == 8
        if simul_set == 9
            control_mode = {'min-rpe'};
        elseif TASK_TYPE == 2020 || TASK_TYPE == 20201 || TASK_TYPE == 20202 || TASK_TYPE == 20203
            control_mode = {'max-rpe','min-rpe'};
            control_mode = {'max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe','max-MB','min-MB','max-MF','min-MF','min-MF-min-MB','max-MF-max-MB','min-MF-max-MB','max-MF-min-MB'};
%             control_mode = {'min-rpe'};
        elseif TASK_TYPE == 2021 || TASK_TYPE == 20211 || TASK_TYPE == 20212 || TASK_TYPE == 20213 || TASK_TYPE == 20214 || TASK_TYPE == 20215
            control_mode = {'max-spe','min-spe'};
            control_mode = {'max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe','max-MB','min-MB','max-MF','min-MF','min-MF-min-MB','max-MF-max-MB','min-MF-max-MB','max-MF-min-MB'};
%             control_mode = {'min-spe'};
        else
            disp('ERROR')
            exit 
%         end
%     elseif TASK_TYPE == 20201 || TASK_TYPE == 20202 || TASK_TYPE == 20203 
%         control_mode = {'max-rpe','min-rpe'};
%     elseif TASK_TYPE == 20211 || TASK_TYPE == 20212 || TASK_TYPE == 20213 || TASK_TYPE == 20214 || TASK_TYPE == 20215 
%         control_mode = {'max-spe','min-spe'};
    end
%     control_mode = {'max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};
%     control_mode = {'max-MB','min-MB','max-MF','min-MF','min-MF-min-MB','max-MF-max-MB','min-MF-max-MB','max-MF-min-MB'};
%     control_mode = {'min-rpe', 'max-rpe'}; %, 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe', 'min-rpe-max-spe'}; 
   % control_mode = {'min-rpe'}; 
    %simul_set = 3;     
    % 0 - nomal, single block training
    % 1 - shuffle_simulation
    % 2 - averaging_simulation
    % 3 - 4 block (1 session) training
    % 4 -
    % 5 -
    % 6 - generate multi subjects
%     FILE_SUFFIX = ['_20210520_' num2str(TASK_TYPE), ''];
%     FILE_SUFFIX = ['_20210616_' num2str(TASK_TYPE), ''];
    %folderpath = '20221021'; % SPE tasks
%     folderpath = '20230627';
    if TASK_TYPE == 2023 % ablation
        folderpath = '20231214'; % Full data
    elseif TASK_TYPE == 2021
%         folderpath = '20230927'; %MB MF rel control
        folderpath = '20231205'; %PE ablation study
        folderpath = '20231212'; %SPE again
%         folderpath = '20231213'; %MB MF rel again
        folderpath = '20231214'; % Full data
%         folderpath = '20240105'; % DDQN P_action data
%         folderpath = '20240201'; % control_task for Rel
%         folderpath = '202405222'; % final figure
        folderpath = '20240717'; % MB MF MPE calc
        control_mode = {'max-MB','min-MB','max-MF','min-MF','min-MF-min-MB','max-MF-max-MB','min-MF-max-MB','max-MF-min-MB'};
    elseif TASK_TYPE == 2020
        folderpath = '20231212'; %RPE again
        folderpath = '20231214'; % Full data
%         folderpath = '20240105'; % DDQN P_action data
%         folderpath = '20240110ab'; % control_task for PE
%         folderpath = '20240201'; % control_task for Rel
%         folderpath = '202405222'; % final figure        
    elseif TASK_TYPE == 20201 || TASK_TYPE == 20202 || TASK_TYPE == 20203
        folderpath = '20240110ab'; % control_task for PE
%         folderpath = '20240201'; % control_task for Rel
        folderpath = '202405222'; % final figure
        folderpath = '20240725'; % Full data
    elseif TASK_TYPE == 20211 || TASK_TYPE == 20212 || TASK_TYPE == 20213 || TASK_TYPE == 20214 || TASK_TYPE == 20215
        folderpath = '20240110ab'; % control_task for PE
%         folderpath = '20240201'; % control_task for Rel
        folderpath = '202405222'; % final figure
        folderpath = '20240725'; % Full data
    elseif TASK_TYPE == 2014
        folderpath = '20231020'; %MB MF rel control + ori task
        folderpath = '20231212'; %ori again
        folderpath = '20231214'; % Full data
    end
    FILE_SUFFIX = ['_' num2str(TASK_TYPE), '_20_trials_' folderpath];
%     FILE_SUFFIX = ['_' num2str(TASK_TYPE), '_lmem2_' folderpath];
    % 20200904 version means 40-or-not based, rpe targeted, flexible fixed, no
    % transitional probability control condition.s
    % TASK_TYPE = 2021;
    
    done=[4,19];
    if simul_set == 7
        job.path0 = '/home/sjh/RPE_pols';
        job.name = 'GRU_model.py';
        for num_time_steps = 1:20
            for num_cells = 2:29
                job.argu = [' --sbj=82 --time=', num2str(num_time_steps), sprintf(' --cells=%d',num_cells),' --epochs=10000 --stop=3'];
                job.pwd = job.path0;    
                job.nth = num_cells*100+num_time_steps;
                JobPython(id,job,'Code',1, rem(num_time_steps,5)+1);
        %             JobPython(id,job,'Code',1);     
                [out] = SubmitJob(id,pw,job);  
            end
        end
    elseif simul_set == 9
        job.path0 = '/home/sjh/RPE_pols';
        fix_con = {'decay', 'non_decay', 'reversal', 'goal','prev'};
        fix_con = {'goal'};
        job.name = 'shuffle_simulation_fixed.py';
        num_env = [200,200,200,200];
        for con_indx = 1:length(fix_con)
            if strcmp(fix_con{con_indx},'prev')
                load('Y:\RPE_pols\SBJ_structure_tot.mat','SBJ2')
                for ii = 0:length(SBJ2)-1
                    for sess_indx = 0:length(SBJ2{1,ii+1}.HIST_behavior_info)-1
                        for blck_indx = 0:floor(length(SBJ2{1,ii+1}.HIST_behavior_info{sess_indx+1})/20)-1                        
                            job.argu = sprintf(' --policy-sbj=%d',ii);
                            job.argu = [job.argu ' --task-type=2021'];
                            job.argu = [job.argu ' --file-suffix=_fixed_con_'];
                            job.argu = [job.argu fix_con{con_indx}];
%                             job.argu = [job.argu '_SPE'];
%                             job.argu = [job.argu '_v2'];
                            job.argu = [job.argu sprintf('_sess_%d',sess_indx)];
                            job.argu = [job.argu sprintf('_blck_%d',blck_indx)];
                            job.argu = [job.argu sprintf(' --sess=%d',sess_indx)];
                            job.argu = [job.argu sprintf(' --blck=%d',blck_indx)];
                            job.pwd = job.path0;    
                            job.nth = ii;
                            JobPython(id,job,'Code',1);
                        %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                        %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                            [out] = SubmitJob(id,pw,job); 
                        end
                    end
                end
            else
                for ii = 0:num_env(con_indx)
                    job.argu = [sprintf(' --policy-sbj=%d',ii) ' --task-type=2021 --file-suffix=_fixed_con_' fix_con{con_indx}];
                    job.argu = [job.argu '_SPE'];
                    job.argu = [job.argu '_v2'];
                    job.pwd = job.path0;    
                    job.nth = ii;
                    JobPython(id,job,'Code',1);
                %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                    [out] = SubmitJob(id,pw,job); 
                end
            end            
        end
    elseif simul_set == 10
        % Recovering rate control space
        job.path0 = '/home/sjh/RPE_pols';
        job.name = 'shuffle_simulation_fixed2.py';
        recover_indx = 0;
        fix_con = {'recover', 'recover2', 'recover3'};
%         fix_con = {'recover2'};
        for con_indx = 1:length(fix_con)
            if strcmp(fix_con{con_indx},'recover')
                for ii = 0:9; for jj = 0:9-ii; recover_indx = recover_indx + (9-ii-jj) + 1; end; end
                for ii = 0:2*recover_indx
                    job.argu = [sprintf(' --policy-sbj=%d',ii) ' --task-type=2020 --file-suffix=_fixed_con_' fix_con{con_indx}];
                    job.argu = [job.argu '_SPE'];
                    job.argu = [job.argu '_v2'];
                    job.pwd = job.path0;    
                    job.nth = ii;
                    JobPython(id,job,'Code',1);
                %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                    [out] = SubmitJob(id,pw,job);     
                    
                    job.argu = [sprintf(' --policy-sbj=%d',ii) ' --task-type=2020 --file-suffix=_fixed_con_' fix_con{con_indx}];
                    job.argu = [job.argu '_v2'];
                    job.pwd = job.path0;    
                    job.nth = ii;
                    JobPython(id,job,'Code',1);
                %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                    [out] = SubmitJob(id,pw,job);  
                end
            else
                recover_indx = 100;
                for ii = 0:2*recover_indx
                    job.argu = [sprintf(' --policy-sbj=%d',ii) ' --task-type=2020 --file-suffix=_fixed_con_' fix_con{con_indx}];
                    job.argu = [job.argu '_SPE'];
                    job.argu = [job.argu '_v2'];
                    job.pwd = job.path0;    
                    job.nth = ii;
                    JobPython(id,job,'Code',1);
                %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                    [out] = SubmitJob(id,pw,job);     
                    
                    job.argu = [sprintf(' --policy-sbj=%d',ii) ' --task-type=2020 --file-suffix=_fixed_con_' fix_con{con_indx}];
                    job.argu = [job.argu '_v2'];
                    job.pwd = job.path0;    
                    job.nth = ii;
                    JobPython(id,job,'Code',1);
                %             JobPython(id,job,'Code',1,rem(ii,6)+1);
                %             JobPython(id,job,'Code',1,5+rem(ii,2));     
                    [out] = SubmitJob(id,pw,job);     
                end
            end
        end
    elseif simul_set == 6
        job.path0 = '/home/sjh/RPE_pols';
        job.name = 'simulation_gen_arbs.py';
        job.argu = [' --episodes=100 --trials=20 ', sprintf(' -n %d',ii),' --file-suffix ', ...
            FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Reproduce_BHV=0', ' --delta-control=0.75'];
        job.pwd = job.path0;    
        job.nth = ii;
        JobPython(id,job,'Code',1,rem(ii,4)+1);
%             JobPython(id,job,'Code',1);     
        [out] = SubmitJob(id,pw,job);  
    elseif simul_set ~= 3
        for con=[1:length(control_mode)]
            job.path0 = '/home/sjh/RPE_pols';
            if simul_set == 0                    
                job.name = 'main.py';   
                job.argu = [' -d --episodes 20000 --trials 20 --ctrl-mode=',control_mode{con}, sprintf(' -n %d',ii),' --disable-detail-plot --file-suffix ', ...
                    FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Reproduce_BHV=0', ' --delta-control=0.75', ' --restore-drop-rate=0.5'];%, ' --mode202010'];
                job.argu = [' -d --episodes 20000 --trials 20 --ctrl-mode=',control_mode{con}, sprintf(' -n %d',ii),' --disable-detail-plot --file-suffix ', ...
                    FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Reproduce_BHV=0', ' --delta-control=0.75', ' --restore-drop-rate=0.00'];%, ' --mode202010'];
%                 job.argu = [job.argu ' --ablation=1'];
                %PMB_CONTROL=0 : no PMB control
                %Reproduce-BHV=1 : Fix dqn and used previously estimated policy
                %to get the behavior data
            elseif simul_set == 1
                job.name = 'shuffle_simulation.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --policy-sbj=%d',ii), sprintf(' --task-type=%d',TASK_TYPE)...
                    ' --file-suffix=' FILE_SUFFIX '_delta_control_highest'];
            elseif simul_set == 2   
                job.name = 'simulation_averaging.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --task-type=%d',TASK_TYPE), ' --file-suffix=_', FILE_SUFFIX ,'_whole_averaging'];
            elseif simul_set == 4
                job.name = 'Data_analysis_new_opt.py';                
                %job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
                job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control_restore_drop --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
                job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control_restore_drop --folderpath=' folderpath, ' --restore-drop-rate=0.0'];
                %job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control_restore_drop --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
%                 job.argu = [job.argu ' --ablation=1'];
            elseif simul_set == 5
                job.name = 'shuffle_simulation_new_opt.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --policy-sbj=%d',ii), sprintf(' --task-type=%d',TASK_TYPE)...
                    ' --file-suffix=' FILE_SUFFIX '_delta_control_restore_drop_highest' ' --restore-drop-rate=0.75'];
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --policy-sbj=%d',ii), sprintf(' --task-type=%d',TASK_TYPE)...
                    ' --file-suffix=' FILE_SUFFIX '_delta_control_restore_drop_highest' ' --restore-drop-rate=0.0'];
            elseif simul_set == 8
                job.name = 'policy_permutation.py';
                
                %job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
                job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control_restore_drop --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
                %job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control_restore_drop --folderpath=' folderpath, ' --restore-drop-rate=0.5'];
%                 job.argu = [job.argu ' --ablation=1'];
%                 job.argu = [job.argu ' --perturbation'];              
                job.argu = [job.argu ' --pseudorandom'];            
            end
            job.pwd = job.path0;    
            job.nth = ii;
            active_cl = [1,2,4,5,6,7,8];
            JobPython(id,job,'Code',1,active_cl(rem(ii,7)+1));
%             JobPython(id,job,'Code',1,rem(ii,6)+1);
%             JobPython(id,job,'Code',1,5+rem(ii,2));     
            [out] = SubmitJob(id,pw,job);   
        end      
    else
        job.path0 = '/home/sjh/RPE_pols';
        sc_sets = perms([1,2,3,4,5]);
        sc_sets = sc_sets(:,1:4);
        %simul_set = 3        
        for sc_indx = 1:length(sc_sets)
            job.name = 'main.py';
            job.argu = [' -d --episodes 1000 --trials 20 --ctrl-mode=',num2str(sc_sets(sc_indx,1)),num2str(sc_sets(sc_indx,2)),num2str(sc_sets(sc_indx,3)),num2str(sc_sets(sc_indx,4)), sprintf(' -n %d',ii),' --disable-detail-plot --file-suffix ', FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Session_block=1'];
            job.pwd = job.path0;
            job.nth = ii*1000+sc_indx;
            JobPython(id,job,'Code',1,rem(ii,6)+1);
            [out] = SubmitJob(id,pw,job);
        end
    end
end
