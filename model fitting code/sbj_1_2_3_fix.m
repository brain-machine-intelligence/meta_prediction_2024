% header='\\143.248.30.94\bmlsamba\SJH_indi\ModelRL\task_2019\result_save';
header='\\143.248.30.94\bmlsamba\SJH_indi\ModelRL\task_2019\result_simul_seed_selected';
LIST_SBJ={'sbj001_SYH','sbj002_OJH','sbj003_HJH','sbj004_KMS'};
for ii=1:4
%     file_name=[header, '\', LIST_SBJ{1,ii}, '_fmri_info_ori.mat'];
%     load(file_name)
%     save([header, '\', LIST_SBJ{1,ii}, '_fmri_info_ori.mat'],'HIST_behavior_info','HIST_behavior_info_Tag','HIST_block_condition','HIST_block_condition_Tag','HIST_event_info','HIST_event_info_Tag','HIST_map_state_info','HIST_map_state_info_Tag')
    for con=1:14
        file_name=[header, '\SBJ_structure_sbj', sprintf('%.2d',ii), '_sjh_con', sprintf('%.2d',con) '.mat'];
        load(file_name)
        save([header, '\SBJ_structure_sbj', sprintf('%.2d',ii), '_sjh_con', sprintf('%.2d',con) '_ori.mat'],'SBJ')
        tot_sess=length(SBJ{1,1}.HIST_block_condition);
        for sess=1:tot_sess
            tmp=SBJ{1,1}.HIST_behavior_info{1,sess}(:,21);
            SBJ{1,1}.HIST_behavior_info{1,sess}(tmp==1,21)=2;
            SBJ{1,1}.HIST_behavior_info{1,sess}(tmp==2,21)=1;
            tmp=SBJ{1,1}.HIST_block_condition{1,sess}(3,:);
            SBJ{1,1}.HIST_block_condition{1,sess}(3,tmp==1)=2;
            SBJ{1,1}.HIST_block_condition{1,sess}(3,tmp==2)=1;
            tmp=SBJ{1,1}.HIST_block_condition{1,sess}(2,:);
            SBJ{1,1}.HIST_block_condition{1,sess}(2,tmp==2)=3;
            SBJ{1,1}.HIST_block_condition{1,sess}(2,tmp==3)=2;
        end
%         save([header, '\', LIST_SBJ{1,ii}, '_fmri_info.mat'],'HIST_behavior_info','HIST_behavior_info_Tag','HIST_block_condition','HIST_block_condition_Tag','HIST_event_info','HIST_event_info_Tag','HIST_map_state_info','HIST_map_state_info_Tag')
        save(filename,'SBJ')
    end
end