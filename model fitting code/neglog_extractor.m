% for con=1:14
%     for ii=1:21
%         disp([con,ii])
%         filename=['\\143.248.30.94\bmlsamba\SJH_indi\Human_Guidance\Behavior_Simul\task_2019\result_simul_seed_selected\SBJ_structure_sbj' sprintf('%.2d',ii) sprintf('_sjh_con%.2d.mat',con)];
%         load(filename)
%         SBJ2(1,ii)=SBJ;
%     end
%     SBJ=SBJ2;
%     save(['\\143.248.30.94\bmlsamba\SJH_indi\Human_Guidance\Behavior_Simul\task_2019\result_simul_seed_selected\total_SBJ_con' sprintf('%.2d.mat',con)],'SBJ');
% end


eval_ArbitrationRL3(param_in, data_in, mode)