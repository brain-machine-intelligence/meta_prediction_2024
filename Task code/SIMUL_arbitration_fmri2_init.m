function [out]=SIMUL_arbitration_fmri2_init(CASE_NUM)
%% relocating all the seed files (from seed_images folder) according to the case #

% CASE_NUM=5; % 1~6

seed_path0=pwd;%['D:\0-program\One shot learning']; % laptop
seed_path=[seed_path0 '\seed\'];
seed_path_pool=[seed_path '\seed_images\'];

% red(1)-blue(2)-yellow(3) color cue mapping for all six cases
case_mapping=[1 2 3; 2 1 3; 3 1 2; 3 2 1; 1 3 2; 2 3 1];

%% relocating state files
state_relo=randperm(5);
for i=1:1:5
    % relocating outcome state files (except outcome states)
    src_file=[seed_path_pool sprintf('s%d.png',state_relo(i))];
    dst_file=[seed_path sprintf('s%03d.png',i)];
    delete(dst_file); WaitSecs(0.7);
    [status,message,messageId]=copyfile(src_file,dst_file,'f'); WaitSecs(0.8);
    if(status~=1)
        disp('-copy error. try to delete files in the seed folder.');
    end
end

%% relocating misc

for i=1:1:3

    % relocating reward message files

    src_file=[seed_path_pool sprintf('case%d_r%d_win.png',CASE_NUM,i)];
    dst_file=[seed_path sprintf('r%03d_win.png',i)];
    delete(dst_file);
    copyfile(src_file,dst_file,'f');    WaitSecs(0.3);

    src_file=[seed_path_pool sprintf('case%d_r%d_lost.png',CASE_NUM,i)];
    dst_file=[seed_path sprintf('r%03d_lost.png',i)];
    delete(dst_file);
    copyfile(src_file,dst_file,'f');    WaitSecs(0.3);

    % relocating goal-box files
    src_file=[seed_path_pool sprintf('g%d.png',case_mapping(CASE_NUM,i))];
    dst_file=[seed_path sprintf('g%03d.png',i)];
    delete(dst_file);
    copyfile(src_file,dst_file,'f');    WaitSecs(0.3);

    % relocating outcome state files
    src_file=[seed_path_pool sprintf('o%d.png',case_mapping(CASE_NUM,i))];
    dst_file=[seed_path sprintf('s%03d.png',5+i)];
    delete(dst_file);
    copyfile(src_file,dst_file,'f');    WaitSecs(0.3);

end

disp('initialization completed.');
out=1;
end