raw_policy=load('opt_pol_2020_p_fix.mat');
raw_policy = raw_policy.pol;
disp('- scheduling start...');
pre_seq=[1,2,3,4; 1,1,1,1]; %First row: type of optimizing, Second row: 1-flexible, 2-specific
%MinSPE MaxSPE MinRPE MaxRPE

%prepare multi-block lists
pre_list = perms(1:5);
pre_list = unique(pre_list(:,1:4),'rows');
sess_opt='fmri'; sess_num=5; name='ksm'; sbj_num = 1; image_num = 1;
trial_length=20; trial_nums=4;
%%%% Expecting 5 main+1 presessions
conditions=[0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. , 0.5, 0. ,  0.5, 0. ,0.5, 0. , 0.5];
conditions=reshape(conditions,[2, length(conditions)/2]);
conditions(1,:)=conditions(1,:)+1;
%%%% This is for fixed scenario simu;
% if strcmp(sess_opt,'pre')
%     init_main_seq_seed = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5];
%     init_main_seq_rand_seed = randperm(5);
%     init_main_seq = [];
%     for ii =1:5
%         init_main_seq = [init_main_seq, randsample(init_main_seq_seed(4*init_main_seq_rand_seed(ii)-3:4*init_main_seq_rand_seed(ii)),4)];
%     end
%     main_seq=zeros(trial_nums,2,5);
%     main_seq(:,2,:)=1;
%     main_seq(:,1,1)=init_main_seq(1:trial_nums);
%     for ii=1:5
%         main_seq(:,1,ii)=init_main_seq(trial_nums*ii-trial_nums+1:trial_nums*ii);
%     end
%     save('main_seq.mat','main_seq');
%     mkdir result_save
% else
load(['init_main_seq_' num2str(sbj_num) '.mat'])
load(['main_seq_' num2str(sbj_num) '.mat'])
% end

%%%%%%%

cons=[];
switch sess_opt
    case 'pre'
        sess_num=1;
        input_scenario=zeros(1,trial_length*trial_nums);
        input_seq=squeeze(raw_policy(1,:))+1;
        for ii=1:trial_nums            
%             input_seq=[input_seq, ones(1,trial_length)*4];
            input_scenario(trial_length*ii-trial_length+1:trial_length*ii)=pre_seq(1,ii);
        end  
        cons=conditions(:,pre_seq(1,1:trial_nums));
    otherwise
        input_scenario=zeros(1,trial_length*trial_nums);
        [~,indx]=ismember(pre_list,init_main_seq(sess_num*4-3:sess_num*4),'rows');
        input_seq=raw_policy(find(indx),:)+1;
        for ii=1:trial_nums
            input_scenario(trial_length*ii-trial_length+1:trial_length*ii)=main_seq(ii,1,sess_num);
            cons=[cons, conditions(:,main_seq(ii,1,sess_num))];
        end
end

input_seq=[input_seq; ones(1,length(input_seq))];

SIMUL_arbitration_fmri_rpe(name, sess_num, sess_opt, image_num, 'real', input_scenario, input_seq, cons);

