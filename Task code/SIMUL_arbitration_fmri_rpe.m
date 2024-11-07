function [output_info]=SIMUL_arbitration_fmri_rpe(PART_NUMBER, SESS_NUMBER, session_opt, COLOR_CASE, test_switch, scenario, seq, cons)
% % % This code is finally updated: 20190826 1am
% [NOTE] the goal is changing in each trial.
% (ex) SIMUL_arbitration_fmri('john',1,'pre')
% (ex) SIMUL_arbitration_fmri('john',1,'fmri')
portSpec = 'COM5';
joker = ''; sampleFreq = 120; baudRate = 115200; specialSettings = [];
InputBufferSize = sampleFreq * 3600;
readTimeout = max(10*1/sampleFreq, 15);
readTimeout = min(readTimeout, 21);
portSettings = sprintf(['%s %s BaudRate=%i InputBufferSize=%i Terminator=0 ReceiveTimeout=%f ' ...
    'ReceiveLatency=0.0001'], joker, specialSettings, baudRate, InputBufferSize, readTimeout);
myport = IOPort('OpenSerialPort', portSpec, portSettings);
asyncSetup = sprintf('%s BlockingBackgroundRead=1 StartBackgroundRead=1',joker);
IOPort('ConfigureSerialPort', myport, asyncSetup);
pktdata = [];
while size(pktdata) > 0
    [pktdata, treceived] = IOPort('Read', myport, 1, 1);
   disp([pktdata, treceived])
end
time_ScanStart = clock;

%IOPort OpenSerialPort?
diary_name = [pwd '/result_save/command_output_' PART_NUMBER '_' session_opt num2str(SESS_NUMBER) '.txt'];
diary diary.txt
diary on
EXP_NAME_IN = num2str(PART_NUMBER); dummy_ii=0;
index_num = SESS_NUMBER; % : session#   
visited_goal_state = 0;

% % % session_opt='fmri'; %'pre','fmri'`

%scanner key device : USB HHSC-2x4-C
% use "key_test.m" to see the actual number of keyboard input.
% check line 320-330 for monitor compatibility
case_num=COLOR_CASE;
current_sess_percent=50;
current_sess_score=0;
%% organizing all the cue presentation files before the practice session
% refer to ../result_save/List_subject_info.xlsx and update!
if(strcmp(session_opt,'pre')==1)
    
    if(index_num==1) % for the first session per each subject
        case_num=COLOR_CASE;
        disp('- now starting to organize the cue presentation files...');
        SIMUL_arbitration_fmri2_init(case_num);
        disp('- organization of the cue presentation files completed.');
    end
end

%% function option check
okay_to_start=0;
if(strcmp(session_opt,'pre')==1)
    okay_to_start=1;
end
if(strcmp(session_opt,'fmri')==1)
    okay_to_start=1;
end
if(okay_to_start~=1)
    error('- ERROR:: check the "session_opt!"');
end

EXP_NAME=[EXP_NAME_IN '_' session_opt];


seed_path0=pwd;%['D:\0-program\One shot learning']; % laptop
seed_path=[seed_path0 '\seed\'];


%% check if session file exists, and other conditions
COND_NEW_FILE=1;
file_chk_name=[EXP_NAME sprintf('_%d.mat',index_num)];
file_name_check=[pwd '\result_save\' file_chk_name];
if(exist(file_name_check)==2)
    disp('$$$$$ ERROR: the corresponding file exists. try another name or session number. $$$$$$$$$$');
    COND_NEW_FILE=0;
    return;
end
if(index_num>1)
    file_chk_name2=[EXP_NAME sprintf('_%d.mat',index_num-1)];
    file_name_check2=[pwd '\result_save\' file_chk_name2];
    if(exist(file_name_check2)==0) % if the previous session file does not exist
        disp('$$$$$ ERROR: The previous session file does not exist. Check the previous session number. $$$$$$$$$$');
        COND_NEW_FILE=0;
        return;
    end
end
MAX_SESSION_NUM=5;
if(index_num>MAX_SESSION_NUM) % MAX session number =5 !
    disp(sprintf('$$$$$ ERROR: exceeds the max # of sessions =%d. $$$$$$$$$$',MAX_SESSION_NUM));
    COND_NEW_FILE=0;
    return;
end

warning('off')
close all
output_info=0;



%% options type1 - display

% screen size - fixed
SCREEN_RESOLUTION=[1280 800];
% SCREEN_RESOLUTION=[1920 1080];

% image size
KEEP_PICTURE_OPT=1; % always 1
DO_TAKE_SNAPSHOT=0;
IMAGE_SIZE=[256 256]; %width,height - for cue presentation
OUTCOME_MSG_SIZE=[800 185]*2; %width,height - for reward message presentation
NUM_MSG_SIZE=[ceil(OUTCOME_MSG_SIZE(1)/15) ceil(OUTCOME_MSG_SIZE(2)/3)]; %width,height - for number presentation
GOAL_IMG_SIZE=[256 245]; % width,height - for goal state presentation
disp_scale=1.0; % for cue presentation
disp_scale_goalimg=0.4; % for goal image presentation
disp_scale_outcome_msg=0.5; % for outcome msg presentation
disp_scale_scoring=0.7; % for mouse-clicking score submission display
Tot_session=1; % # of total sessions (fixed because this runs for each session)

% text size
text_size_default=20; % font size (don't change)
text_size_reward=800; % height in pixel
use_unicode = true; %false;

% background color
BackgroundColor_block_intro=[130,130,130,150]; % gray
BackgroundColor_Cue_page=[210,210,210,150]; % light gray
BackgroundColor_Trial_ready_page=[210,210,210,150]; % light gray
BackgroundColor_Reward_page=[210,210,210,150]; % light gray
COLOR_FIXATION_MARK=[70,70,70,200]; % dark gray


switch test_switch
    
    case 'test'
        
        % pre-session : # of blocks
        if(strcmp(session_opt,'pre')==1)
            Tot_block=4;%length(seq(1,:)); % # of total blocks (MUST be the multitude of 4(=#of conditions)
        end
        % fmri-session : # of blocks
        if(strcmp(session_opt,'fmri')==1)
            Tot_block=4;%length(seq(1,:)); % 16; % # of total blocks (MUST be the multitude of 4(=#of conditions)
        end
        
    case 'real'
        Tot_block = 4;%length(seq(1,:));
               
    otherwise
        
        error('problem with test_switch specification');
        
end


%% options type2 - display speed, trial length, ITI, timing,...

% session schedule
        % G: specific-goal, state transition probability=(0.9,0.1):low uncertainty 
        % G': specific-goal, state transition probability=(0.5,0.5):high uncertainty
        % H: flexible-goal, state transition probability=(0.5,0.5):high uncertainty 
        % H': flexible-goal, state transition probability=(0.9,0.1):low uncertainty
range_num_trials_G0_G1_H0_H1=[[3,5];[5,7];[5,7];[3,5]]; % each row: minmax # of trials of G,G',H,H'
time_estimation_trial_sec=6; %(sec)- used to estimate session time when scheduling
time_limit_session_min=30; %(min) - rescheduling until the time estimation meets thie criterion (limit)
ffw_speed = 1.5;

% sec_stim_ready=.1; %(sec)
% sec_trial_ready=1.0/ffw_speed; %(sec)
sec_scanner_ready=5/ffw_speed; % sec for scanner stabilization
% sec_block_ready=0.5/ffw_speed; % sec for block ready signal
% sec_stim_display=0.0/ffw_speed;
sec_stim_interval=[1 4]/(ffw_speed);%1.5; %(sec)
%%% sec_trial_interval=[1 1]/(ffw_speed);%1.5; %(sec) %%% SFW: is unused
sec_limit_decision=4;%/(ffw_speed); % time limit of decision (sec)
sec_jittered_blank_page=0.15/(ffw_speed); % (sec)
sec_reward_display=2/ffw_speed; % only for 'fmri' option. 1.5sec for 'pre' session



%% key code definition

% fMRI scanner system setting
%KEY_L=65;
%KEY_R=68;
%KEY_Y=51;
%KEY_N=52;
%KEY_Q=81;
%KEY_T=53;
%KEY_S=83;

% my laptop setting
KEY_L=49;%KbName('a');%37; %a
KEY_R=51;%KbName('d');%39; %d
KEY_Y=KbName('y'); %'y'
KEY_N=KbName('n'); %'n'
KEY_Q=KbName('q'); %'q'
KEY_T=KbName('t'); % 't', 5 in desktop, 84 in laptop
KEY_S=53;%KbName('b'); % 'b'
KEY_B=50;%KbName('b'); % 'b'
KEY_C=52;%KbName('c'); % 'b'
trigger_key=53;%KbName('space'); %'s'
%  KEY_L=65;%37; %a
%  KEY_R=68;%39; %d
%  KEY_Y=89; %'y'
%  KEY_N=78; %'n'
%  KEY_Q=81; %'q'
%  KEY_T=84; % 't', 5 in desktop, 84 in laptop
%  KEY_S=66; % 'b'
%  trigger_key=83; %'s'






%%
%% INITIALIZATION STARTS HERE
%%





%% creating the mother map and state
map_opt.transition_prob_seed=[0.5 0.5];%[0.9 0.1];
map_opt.reward_seed=[40 20 10 0];
if strcmp(session_opt,'pre')
    [myMap N_state N_action N_transition]=Model_Map_Init2('jaehoon2020',map_opt);
    save('maps.mat','myMap','N_state','N_action','N_transition')
else
    load('maps.mat','myMap','N_state','N_action','N_transition')
end
map_sbj=myMap;
% create my state
myState=Model_RL_Init(N_state,N_action,N_transition);


%% global variables (for each block)
HIST_event_info0=[];
HIST_behavior_info0=[];
if(index_num==1) % create the image usage matrix if this is the first session

    % event   : HIST_event_info{1,session#}
    HIST_event_info_Tag{1,1}='row1 - block#';    HIST_event_info_Tag{2,1}='row2 - trial# (in each block), 0 if outside of the trial';     HIST_event_info_Tag{3,1}='row3 - trial_s#, 0 if outside of the trial_s';
    HIST_event_info_Tag{4,1}='row4 - event time in session';   HIST_event_info_Tag{5,1}='row5 - event time in block';       HIST_event_info_Tag{6,1}='row6 - event time in trial';
    HIST_event_info_Tag{7,1}='row7 - state. 0.5: fixation mark on, 1: S1, 2: S2, 3: S3, 4: S4, 5: S5, 6(+/-)0.1: O1(with win/lost msg), 7(+/-)0.1: O2(with win/lost msg), 8(+/-)0.1: O3(with win/lost msg), 9: O4, 10:A1, 11:A2, 20: a short blank page display, -99:fail to choose in time limit, (-) when display off';
    HIST_event_info_Tag{8,1}='row8 - goal state=outcome state. 6: 40reward state, 7: 20reward state, 8: 10reward state, -1: universal state';

    % behavior : HIST_behavior_info{1,session#}    
    HIST_behavior_info_Tag{1,1}='col1 - block #';    HIST_behavior_info_Tag{1,2}='col2 - trial # (in each block)';
    %HIST_behavior_info_Tag{1,3}='col3 - block condition - 1: G(with low uncertainty), 2: G''(with high uncertainty), 3:H(with high uncertainty), 4:H''(with low uncertainty)';
    HIST_behavior_info_Tag{1,3}='col3 - 1st action policy - 1: defualt, 2: visited recovery, 3: unvisited recovery';
    %1: default, 2: deterministic, 3: stochastic, 4: re-distribution, 5 %reset'
    HIST_behavior_info_Tag{1,4}='col4 - currently dummy/2nd action policy - currently no meaning';
    %1: default, 2: deterministic, 3: stochastic, 4: re-distribution, 5: reset';
    HIST_behavior_info_Tag{1,5}='col5 - S1';
    HIST_behavior_info_Tag{1,6}='col6 - S2';        
    HIST_behavior_info_Tag{1,7}='col7 - S3';
    HIST_behavior_info_Tag{1,8}='col8 - A1 (action in S1)';     
    HIST_behavior_info_Tag{1,9}='col9 - A2 (action in S2)'; 
    HIST_behavior_info_Tag{1,10}='col10 - RT(A1)';        
    HIST_behavior_info_Tag{1,11}='col11 - RT(A2)';
    HIST_behavior_info_Tag{1,12}='col12 - on set (S1) from the trial start';      
    HIST_behavior_info_Tag{1,13}='col13 - onset (S2) from the trial start';      
    HIST_behavior_info_Tag{1,14}='col14 - onset (S3) from the trial start';
    HIST_behavior_info_Tag{1,15}='col15 - onset (A1) from the trial start';      
    HIST_behavior_info_Tag{1,16}='col16 - onset (A2) from the trial start';
    HIST_behavior_info_Tag{1,17}='col17 - reward amount (0/10/20/40) at S3';
    HIST_behavior_info_Tag{1,18}='col18 - total amount (total in the current session)';
    HIST_behavior_info_Tag{1,19}='col19 - goal state=outcome state. 6: 40reward state, 7: 20reward state, 8: 10reward state, -1: universal state';
    HIST_behavior_info_Tag{1,20}='col20 - block condition - 1: min RPE, 2: max RPE, 3: min SPE, 4: max SPE. 5: baseline';
    %'col20 - block condition - 1: min SPE, 2: max SPE, 3: min RPE, 4: max RPE. 5: baseline' ;%'col20 -block condition - 0: max SPE, 1: min SPE, 2: max RPE, 3: min RPE. 4: max SPE max RPE, 5: max SPE min RPE, 6: min SPE max RPE, 7: min SPE max RPE';
    HIST_behavior_info_Tag{1,21}='dummy'; %'col21 - block condition - 1: flexible goal, 2: specific goal';
    HIST_behavior_info_Tag{1,22}='col22 - block confidence';
    HIST_behavior_info_Tag{1,23}='col23 - dummy score';
    HIST_behavior_info_Tag{1,24}='col24 - dummy ranking';
    
    HIST_map_state_info_Tag{1,1}='state - state used in the s1,s2,s3';
    HIST_map_state_info_Tag{1,2}='map - map used in the s1,s2,s3';

    
else % load image usage matrix to update
    file_imgind_ld_name=[EXP_NAME '_info.mat'];
    file_name_ld=[pwd '\result_save\' file_imgind_ld_name];
    load(file_name_ld);
end

% seed image read
img_set=cell(1,9); % including othe output states
for img_ind=1:1:9
    file_full_path=[seed_path sprintf('s%03d.png',img_ind)];
    img_set{1,img_ind}=imresize(imread(file_full_path),[ IMAGE_SIZE(1) IMAGE_SIZE(2) ]); % get a frame %%% TURNED AROUND BY SFW (new imresize file)
end
% reward message image read
img_set_reward_msg_win=cell(1,4); % msg for 40, 20, 10, 0
img_set_reward_msg_lost=cell(1,3); % msg for 40, 20, 10
for jj=1:1:4
    file_full_path=[seed_path sprintf('r%03d_win.png',jj)];
    img_set_reward_msg_win{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
end
for jj=1:1:3
    file_full_path=[seed_path sprintf('r%03d_lost.png',jj)];
    img_set_reward_msg_lost{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
end
% img_set_reward_msg_win2=cell(6,4); % msg for 40, 20, 10, 0
% img_set_reward_msg_lost2=cell(6,3); % msg for 40, 20, 10
% for ii=1:1:6
%     for jj=1:1:3
%         file_full_path=[seed_path sprintf('seed_images2/case%d_r%d_win.png',ii,jj)];
%         img_set_reward_msg_win2{ii,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
%     end
%     file_full_path=[seed_path 'seed_images2/r004_win.png'];
%     img_set_reward_msg_win2{ii,4}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
% end
for ii=1:6
    for jj=1:1:3
        file_full_path=[seed_path sprintf('seed_images2/case%d_r%d_lost.png',ii,jj)];
        img_set_reward_msg_lost2{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
    end
end
% goal state image read
img_set_goal=cell(1,4); % 40, 20 ,10, universal
for jj=1:1:4
    file_full_path=[seed_path sprintf('g%03d.png',jj)];
    % img_set_goal{1,jj}=imresize(imread(file_full_path),[GOAL_IMG_SIZE(2) GOAL_IMG_SIZE(1)]); %%% SKIPPED RESIZING BY SFW
    img_set_goal{1,jj}=imread(file_full_path);
end

%msg image read
img_set_msg = cell(1,8);
for jj=1:1:8
    file_full_path=[pwd '/images/' sprintf('%04d.jpg',jj)];
    img_set_msg{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
end

tmp = size(img_set_msg{1,1});

%num image read
img_set_num = cell(1,11);
for jj=0:1:9
    file_full_path=[pwd '/images/' sprintf('%d.jpg',jj)];
    img_set_num{1,jj+1}=imresize(imread(file_full_path),[ceil(tmp(1)/3) NaN]); %%% TURNED AROUND BY SFW (new imresize file)
end
img_set_num{1,11}=imresize(imread([pwd '/images/..jpg']),[ceil(tmp(1)/3) NaN]); %%% TURNED AROUND BY SFW (new imresize file)



%% Scheduling (block sequence)
% G:1, G':2, H:3, H':4 
% # of trials : G,H'~[3,5], G',H~[5,7]
criterion=0;
HIST_block_condition_Tag{1,1}='row1: block #';
%HIST_block_condition_Tag{2,1}='row2: block condition of each trial: 1:G, 2:G'', 3:H, 4:H''';
HIST_block_condition_Tag{2,1}='row2: action policy - 1: defualt, 2: visited recovery, 3: unvisited recovery';
% 1: default, 2: deterministic, 3: stochastic, 4: re-distribution, 5: reset';
HIST_block_condition_Tag{3,1}='row3: block condition - block condition - 1: min RPE, 2: max RPE, 3: min SPE, 4: max SPE. 5: baseline'; %0: max SPE, 1: min SPE, 2: max RPE, 3: min RPE. 4: max SPE max RPE, 5: max SPE min RPE, 6: min SPE max RPE, 7: min SPE max RPE';
HIST_block_condition_Tag{4,1}='row4: - block condition - 1: flexible goal, 2: specific goal';
% HIST_block_condition_Tag{5,1}='row5 - block condition - 1: 0.5/0.5, 2: 0.9/0.1';
HIST_block_condition_Tag{5,1}='row5 - visited_goal_state';
disp('- scheduling start...');
single_block_condition=[]; % (2xtrial#) row1: block#, row2: the block condition of each trial
trial_length=20;
single_block_condition=zeros(5,length(seq(1,:)));
for ii=1:4
    single_block_condition(1,trial_length*(ii-1)+1:trial_length*ii)=ii;%HIST_block_condition_Tag{1,1}='row1: block #';
end
single_block_condition(2,:)=seq(1,:);%HIST_block_condition_Tag{2,1}='row2: action policy - 1: default, 2: deterministic, 3: stochastic, 4: re-distribution, 5: reset';
single_block_condition(3,:)=scenario;%HIST_block_condition_Tag{3,1}='row3: block condition -  1: min RPE, 2: max RPE, 3: min SPE, 4: max SPE. 5: baseline'; 1: min SPE, 2: max SPE, 3: min RPE, 4: max  RPE. 5: min SPE min RPE, 6: max SPE max RPE, 7: min SPE max RPE, 8: min SPE max RPE';
single_block_condition(4,:)=1;%HIST_block_condition_Tag{4,1}='row4 - block condition - 1: flexible goal, 2: specific goal'
%single_block_condition(5,:)=1;%HIST_block_condition_Tag{5,1}='row5 - block condition - 1: 0.5/0.5, 2: 0.9/0.1'
single_block_condition(5,:)=0;%HIST_block_condition_Tag{5,1}='row5 - visited_goal_state'
HIST_block_condition{1,index_num}=single_block_condition;
disp('- proceed to the experiment...');





%%
%% EXPERIMENT STARTS HERE
%%


%% Display initialization
whichScreen = 0;
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference','TextRenderer', whichScreen);
wPtr  = Screen('OpenWindow',whichScreen);
[screenWidth, screenHeight] = Screen('WindowSize', wPtr);
HideCursor;

white = WhiteIndex(wPtr); % pixel value for white
black = BlackIndex(wPtr); % pixel value for black
gray = (white+black)/2;
inc = white-gray;
inc_0=white-black;

imageArray={};

% Translate origin into the geometric center of text:
Screen('glTranslate', wPtr, screenWidth/2, screenHeight/2, 0);

% Apple a scaling transform which flips the diretion of x-Axis,
% thereby mirroring the drawn text horizontally:
Screen('glScale', wPtr, -1, -1, 1);

% We need to undo the translations...
Screen('glTranslate', wPtr, -screenWidth/2, -screenHeight/2, 0);


%% starting message
Screen('TextSize',wPtr, text_size_default);
unicode_txt=[39, 49892, 54744, 51060, 32, 44263, 32, 49884, 51089, 46121, 45768, 45796, 46, 32, 92, 110, 32, 92, 110, 32, 40, 49892, 54744, 51088, 51032, 32, 50836, 52397, 50640, 46384, 46972, 32, 50812, 51901, 51060, 45208, 32, 50724, 47480, 51901, 32, 48260, 53948, 51012, 32, 45580, 47084, 51452, 49464, 50836, 46, 32, 41, 39];
if use_unicode
%     DrawFormattedText(wPtr, unicode_txt, 'center', 'center');
    %     DrawFormattedText(wPtr, unicode2native('실험이 곧 시작됩니다. \n (실험자의 요청에따라 왼쪽이나 오른쪽 버튼을 눌러주세요. )','UTF-16'), 'center', 'center');
    %     DrawFormattedText(wPtr, 'click if you ready', 'center', 'center');
    input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,1});
    %
    % (3) add outcome message
    xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
    sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
    sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
    xpos2=xpos; ypos2=ypos;%+sy/2+sy2/2+50;
    destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
    % (4) display on
    Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
else
    DrawFormattedText(wPtr, unicode2native('실험이 곧 시작됩니다. \n (실험자의 요청에따라 왼쪽이나 오른쪽 버튼을 눌러주세요. )','UTF-8'), 'center', 'center');
end
%DrawFormattedText(wPtr, 'The experiment will start soon...\n(Please press Left or Right to start, when instructed)', 'center', 'center');
Screen('Flip', wPtr);  
% KbWait; % temporarily disabled for test APR 21
treceived = 0; pktdata = [];
while treceived == 0
    [pktdata, treceived] = IOPort('Read', myport, 1, 1); %pktdata를 kbcheck의 keycode처럼 쓰면 됩니다
    pktdata = ceil(pktdata);
    %[ keyIsDown, Secs, keyCode ] = KbCheck; % or KbCheck([-1]) for checking from all devices
    if treceived > 0 %keyIsDown
        %[tmp, tmp_key_code]=find(keyCode==1);
        if ismember(KEY_L,pktdata) || ismember(KEY_R,pktdata) || ismember(KEY_B,pktdata) || ismember(KEY_C,pktdata)  % go sign
        %if ismember(KEY_L,tmp_key_code) || ismember(KEY_R,tmp_key_code)  % go sign
%         if keyIsDown
                     break;
        end
        % If the user holds down a key, KbCheck will report multiple events.
        % To condense multiple 'keyDown' events into a single event, we wait until all
        % keys have been released.
%       while KbCheck; end
    end
 end


if(DO_TAKE_SNAPSHOT==1)
    snapshot=Screen(wPtr, 'GetImage', [1, 1, floor(1.0*SCREEN_RESOLUTION(1)), floor(1.0*SCREEN_RESOLUTION(2))]);
    imageArray=[imageArray; {snapshot}];
end


%% waiting for the trigger sign from the scanner
unicode_txt = [49884, 51089, 32, 49888, 54840, 47484, 32, 44592, 45796, 47532, 44256, 32, 51080, 49845, 45768, 45796, 46, 46, 46, 32, 92, 110, 32, 92, 110, 32, 51104, 44624, 32, 44592, 45796, 47140, 51452, 49464, 50836, 46];
if use_unicode 
%     DrawFormattedText(wPtr, unicode_txt, 'center', 'center')
    input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,2});
    %
    % (3) add outcome message
    xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
    sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
    sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
    xpos2=xpos; ypos2=ypos;%+sy/2+sy2/2+50;
    destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
    % (4) display on
    Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
else
    DrawFormattedText(wPtr, '신호를 기다리고 있습니다.', 'center', 'center')
end
%DrawFormattedText(wPtr, 'Now waiting for the trigger...', 'center', 'center')
 Screen('Flip',wPtr);
 WaitSecs(0.1);
 
 %Look for trigger pulse
 treceived = 0; pktdata = [];
% while 1
while treceived == 0
    [pktdata, treceived] = IOPort('Read', myport, 1, 1); %pktdata를 kbcheck의 keycode처럼 쓰면 됩니다
    %[ keyIsDown, Secs, keyCode ] = KbCheck; % or KbCheck([-1]) for checking from all devices
    if treceived > 0 %keyIsDown
        %[tmp, tmp_key_code]=find(keyCode==1);
        if ismember(KEY_S,pktdata)  % go sign
%         if ismember(KEY_L,pktdata) || ismember(KEY_R,pktdata) || ismember(KEY_B,pktdata) || ismember(KEY_C,pktdata)  % go sign
                    break;
        end
        % If the user holds down a key, KbCheck will report multiple events.
        % To condense multiple 'keyDown' events into a single event, we wait until all
        % keys have been released.
%       while KbCheck; end
    end
 end


%% clock-start and then wait for another 5secs until the scanner stabilizes
session_clock_start = GetSecs;
WaitSecs(sec_scanner_ready);


%% block
tot_indx=0;
for block=1:1:Tot_block % each block
    visited_goal_state = 0;
    %% I. Stimuli presentation
    zzz=[];
    block_clock_start = GetSecs;
    for img_ind=1:1:9
        file_full_path=[seed_path sprintf('s%03d.png',img_ind)];
        img_set{1,img_ind}=imresize(imread(file_full_path),[ IMAGE_SIZE(1) IMAGE_SIZE(2) ]); % get a frame %%% TURNED AROUND BY SFW (new imresize file)
    end
    for jj=1:1:4
        file_full_path=[seed_path sprintf('r%03d_win.png',jj)];
        img_set_reward_msg_win{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
    end
    for jj=1:1:3
        file_full_path=[seed_path sprintf('r%03d_lost.png',jj)];
        img_set_reward_msg_lost{1,jj}=imresize(imread(file_full_path),[ NaN text_size_reward ]); %%% TURNED AROUND BY SFW (new imresize file)
    end
    for jj=1:1:4
        file_full_path=[seed_path sprintf('g%03d.png',jj)];
        img_set_goal{1,jj}=imread(file_full_path);
    end

    trial_in_block=0;
    temp=HIST_block_condition{1,index_num};
    [tmp, trial_set]=find(temp(1,:)==block);
%     [myMap, N_state, N_action, N_transition]=Model_Map_Init2('jaehoon2020',map_opt);
    map_sbj.reward=map_sbj.reward_save;
    map_sbj.visited_goal_state = 1;
    map_sbj_update=struct();
    state_sbj_update=struct();
    trial_indx=0;
    temp_map_reward=map_sbj.reward;
    
    for trial=trial_set % each trial
        tot_indx=tot_indx+1;
        trial_indx=trial_indx+1;
        action_indx=0;
        state_sbj=myState;
        map_sbj.index=1;
        state_sbj.index=1;
        map_sbj.trial=trial;
        trial_in_block=trial_in_block+1;
        trial_clock_start = GetSecs;

        onset_state_from_trial_start=[];
        onset_action_from_trial_start=[];
        %current_goal_state=randperm(3,1)+5;
        current_goal_state=-1;


        %% [1] map preparation (G:1, G':2, H:3, H':4)
        % G: specific-goal, state transition probability=(0.9,0.1) 
        % G': specific-goal, state transition probability=(0.5,0.5)
        % H: flexible-goal, state transition probability=(0.5,0.5) 
        % H': flexible-goal, state transition probability=(0.9,0.1) 
        block_condition=[single_block_condition(2,tot_indx); single_block_condition(2,tot_indx)];
%         trial_in_block_condition=policy();
        prob_seed_mat=[0.5 0.5];
        trial_in_block_condition=block_condition(2); 
        
        if rem(trial,trial_length)==1
            map_sbj.reward=map_sbj.reward_save;
        else
            single_block_condition(4,trial)=cons(1,block);
            map_opt.transition_prob_seed=[0.5 0.5];
            single_block_condition(5,trial)=visited_goal_state;
        end
        if visited_goal_state > 0
           map_sbj.reward(visited_goal_state) = map_sbj.reward(visited_goal_state)*(0.7+0.2*rand);
           map_sbj.reward(visited_goal_state) = ceil(10*map_sbj.reward(visited_goal_state))/10;
           if map_sbj.reward(visited_goal_state) < 1
               map_sbj.reward(visited_goal_state) = 1;
           end
        end
        switch trial_in_block_condition
            case 1  % default
                % set T_prob\
                % set Reward
                current_goal_state=-1;
                %map_sbj.reward=map_sbj.reward_save;
                %map_sbj.reward=zeros(N_state,1);
                %map_sbj.reward(current_goal_state)=map_sbj.reward_save(current_goal_state);

            case 2 % R-recovery (Visited)
                % set T_prob
                if visited_goal_state > 0
                    map_sbj.reward(visited_goal_state) = map_sbj.reward(visited_goal_state) * (1.25 + 0.25*rand);
                    map_sbj.reward(visited_goal_state) = ceil(10*map_sbj.reward(visited_goal_state))/10;
                    if map_sbj.reward(visited_goal_state) > map_sbj.reward_save(visited_goal_state)
                        map_sbj.reward(visited_goal_state) = map_sbj.reward_save(visited_goal_state);
                    end
                end
                single_block_condition(5,trial)=visited_goal_state;
                HIST_block_condition{1,index_num}=single_block_condition;
                % set Reward
                % map_sbj.reward=map_sbj.reward_save; % all the rwd state intact                            

            case 3 % R-recovery (unvisited)
                % set T_prob
                HIST_block_condition{1,index_num}=single_block_condition;
                if visited_goal_state > 0
                    for ii = 1:length(map_sbj.reward)
                        if ii~=visited_goal_state
                            map_sbj.reward(ii)=map_sbj.reward(ii)*(1.125+0.25*rand);
                            map_sbj.reward(ii)=ceil(10*map_sbj.reward(ii))/10;
                            if map_sbj.reward(ii) > map_sbj.reward_save(ii)
                                map_sbj.reward(ii) = map_sbj.reward_save(ii);
                            end
                        end
                    end
                end
                % set Reward
                % map_sbj.reward=map_sbj.reward_save; % all the rwd state intact
        end
        
%         disp(sprintf('line:509, trial=%d, policy=%d, FS=%d, HL=%f',trial,trial_in_block_condition,single_block_condition(4,trial), map_opt.transition_prob_seed(1)));                
        disp(sprintf('line:562, trial=%d, scenario=%d, policy=%d', trial,single_block_condition(3,trial),trial_in_block_condition(1)));                
        
        while(map_sbj.IsTerminal(state_sbj.state_history(state_sbj.index))==0)

            current_state=state_sbj.state_history(state_sbj.index);

            map_sbj_update.map_sbj(map_sbj.index)=map_sbj;
            state_sbj_update.state_sbj(state_sbj.index)=state_sbj;
            %% [2] display
            % (0) basic setting
            % T_prob encoding to the current map
            for mm=1:1:2
                for nn=1:1:size(map_sbj.connection_info{1,mm},1)
                    map_sbj.action(1,mm).prob(nn,map_sbj.connection_info{1,mm}(nn,:))=prob_seed_mat;
                end
                map_sbj.action(1,mm).connection=double(map_sbj.action(1,mm).prob&ones(N_state,N_state));
            end            
            disp(['line 637: state - ' num2str(current_state) ',action 1 - state ' num2str(find(map_sbj.action(1,1).connection(current_state,:))) ', action 2 -state ' num2str(find(map_sbj.action(1,2).connection(current_state,:)))])
            % (1) display fixation mark and display off during the jittered interval
            Screen('FillRect',wPtr,BackgroundColor_Cue_page);
            if(current_state==1)
%                   DrawFormattedText(wPtr, [51456, 48708], 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.             
%                 DrawFormattedText(wPtr, 'ready', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
                input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,3});
                %
                % (3) add outcome message
                xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
                sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
                sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
                xpos2=xpos; ypos2=ypos;%+sy/2+sy2/2+50;
                destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
                % (4) display on
                Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
            else
                DrawFormattedText(wPtr, '.', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
            end
            Screen(wPtr, 'Flip');
            HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
                (GetSecs-session_clock_start); (GetSecs-block_clock_start); (GetSecs-trial_clock_start); 0.5; current_goal_state]]; % event save
            sec_stim_interval0=rand*(max(sec_stim_interval)-min(sec_stim_interval))+min(sec_stim_interval);
            WaitSecs(sec_stim_interval0);
            
            % (2-1) add state image
            Screen('FillRect',wPtr,BackgroundColor_Cue_page);
            input_stim = Screen('MakeTexture', wPtr, img_set{1,current_state});
            xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
            sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
            destrect=[xpos-sx/2,ypos-sy/2,xpos+sx/2,ypos+sy/2];            
            Screen('DrawTexture', wPtr, input_stim,[],destrect);

            % (2-2) add goal image            
%             if(current_goal_state==-1) % universal state
%                 current_goal_state_ind=4;
%             else % other goal states 
%                 current_goal_state_ind=current_goal_state-5;
%             end
%             input_stim2 = Screen('MakeTexture', wPtr, img_set_goal{1,current_goal_state_ind});
%             sx2=floor(GOAL_IMG_SIZE(1)*disp_scale_goalimg);       sy2=floor(IMAGE_SIZE(2)*disp_scale_goalimg);
%             xpos2=xpos; ypos2=ypos+sy/2+sy2/2+50;
%             destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];            
%             Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
            
            % (2-3) display on
            Screen(wPtr, 'Flip'); % display on
            clock_time_limit_start=GetSecs;
            onset_state_from_trial_start=[onset_state_from_trial_start (GetSecs-trial_clock_start)];
            HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
                (GetSecs-session_clock_start); (GetSecs-block_clock_start); onset_state_from_trial_start(end); current_state; current_goal_state]]; % event save

            %% [3] get chioce and update            
            decision_made=0;
            WaitSecs(0.1);
            pktdata = [];
            while(~decision_made)
%                 [secs, keyCode] = KbPressWait([], clock_time_limit_start+sec_limit_decision); % if no keyboard in time limit, then go ahead. if pressed earlier, then go ahead.
                [pktdata,treceived] = IOPort('Read',myport,1,1)
                pktdata = ceil(pktdata);
%                 [ keyIsDown, secs, keyCode ] = KbCheck; % or KbCheck([-1]) for checking from all devices
                onset_action_from_trial_start=[onset_action_from_trial_start (GetSecs-trial_clock_start)]; 
                state_sbj.RT(state_sbj.index)=GetSecs-clock_time_limit_start;                
                %[tmp tmp_key_code]=find(keyCode==1);         
                if ismember(KEY_L,pktdata) || ismember(KEY_B,pktdata)%tmp_key_code) % L pressed
                    state_sbj.action_history(state_sbj.index)=1;
                    decision_made=1;
                end
                if ismember(KEY_R,pktdata)|| ismember(KEY_C,pktdata)%tmp_key_code) % R pressed
                    state_sbj.action_history(state_sbj.index)=2;
                    decision_made=1;
                end
                if ismember(KEY_L,pktdata) && ismember(KEY_R,pktdata) && ismember(KEY_B,pktdata) && ismember(KEY_C,pktdata)
                    beep;
                end
%                 if ismember(KEY_Q,tmp_key_code) % 'q' pressed for aborting
%                     state_sbj.action_history(state_sbj.index)=ceil(2*rand); % random select if fail to make a decision
%                     HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
%                         (GetSecs-session_clock_start); (GetSecs-block_clock_start); onset_action_from_trial_start(end); -99; current_goal_state]]; % event save
%                     qwe;
%                     break;
%                     decision_made=1;                    
%                 end                
                % check the time limit !@#$                
                if((state_sbj.RT(state_sbj.index)>sec_limit_decision)&&(decision_made==0)) % no decision made in time limit
                    state_sbj.action_history(state_sbj.index)=ceil(2*rand); % random select if failed to make a decision in time
                    decision_made=1;            Is_bet=1;
                    HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
                        (GetSecs-session_clock_start); (GetSecs-block_clock_start); onset_action_from_trial_start(end); -99; current_goal_state]]; % event save
                else % in time, and decision made
                    if(decision_made==1)
                        HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
                            (GetSecs-session_clock_start); (GetSecs-block_clock_start); onset_action_from_trial_start(end); 9+state_sbj.action_history(state_sbj.index); current_goal_state]]; % event save
                    end
                end
            end
            
            %% [3] moving to the next state
            [state_sbj map_sbj]=StateSpace(state_sbj,map_sbj);  % map&state index ++
            
            
            

        end % end of each choice
        
        
        %% [4] terminal state: display reward
        current_state=state_sbj.state_history(state_sbj.index);
        
        % (0) display fixation mark and display off during the jittered interval
        Screen('FillRect',wPtr,BackgroundColor_Cue_page);
        DrawFormattedText(wPtr, '.', 'center', 'center', COLOR_FIXATION_MARK); % add 'o' mark at the click pt.
        Screen(wPtr, 'Flip');
        HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
            (GetSecs-session_clock_start); (GetSecs-block_clock_start); (GetSecs-trial_clock_start); 0.5; current_goal_state]]; % event save
        sec_stim_interval0=rand*(max(sec_stim_interval)-min(sec_stim_interval))+min(sec_stim_interval);
        WaitSecs(sec_stim_interval0);
        
        % (1) add state image
        Screen('FillRect',wPtr,BackgroundColor_Cue_page);
        input_stim = Screen('MakeTexture', wPtr, img_set{1,current_state});
        xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
        sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
        destrect=[xpos-sx/2,ypos-sy/2,xpos+sx/2,ypos+sy/2];
        Screen('DrawTexture', wPtr, input_stim,[],destrect);

        % (2) outcome message read
        outcome_state=state_sbj.state_history(3);
        original_rwd=map_sbj.reward_save(outcome_state);
%         actual_rwd=state_sbj.reward_history(3);
        target_rwd=state_sbj.reward_history(3);                
        visited_goal_state = outcome_state;
        case_earn=1;
        
        % msg for 40, 20, 10, 0 added to the total                
        input_stim2 = Screen('MakeTexture', wPtr, img_set_goal{1,visited_goal_state-5});
%            
        % (3) add outcome message
        sx2=floor(GOAL_IMG_SIZE(1)*disp_scale_goalimg);       sy2=floor(IMAGE_SIZE(2)*disp_scale_goalimg);
        xpos2=xpos; ypos2=ypos+sy/2+sy2/2+50;
        destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
        % (4) display on        
        Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
        reward_display = sprintf('%.1f',map_sbj.reward(visited_goal_state));
        unicode_display = [];
        for unicode_indx = 1:length(reward_display)
            if strcmp(reward_display(unicode_indx),'.')
                unicode_display = [unicode_display 46];
            else
                unicode_display = [unicode_display ceil(str2num(reward_display(unicode_indx))+48)];
            end
        end
        unicode_display = [unicode_display 32, 46041, 51204, 51012, 32, 54925, 46301, 54664, 49845, 45768, 45796, 46];
        input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,4});
        %
        % (3) add outcome message
        xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
        sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
        sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
        xpos2=xpos; ypos2=destrect2(4)+sy2/2+50;
        destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
        % (4) display on
        Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
        
%         DrawFormattedText(wPtr, [ sprintf('%.1f',map_sbj.reward(visited_goal_state))], destrect2(1), (destrect2(2)+destrect2(4))/2);%,'','',1);    
        
        tmp_1st_digit = floor(map_sbj.reward(visited_goal_state)/10);
        input_stim3 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_1st_digit+1});
        sx3=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy3=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
        xpos3=xpos - 7*sx3 - 1; ypos3=ypos2;
        destrect3=[xpos3-sx3/2,ypos3-sy3/2,xpos3+sx3/2,ypos3+sy3/2];
        Screen('DrawTexture', wPtr, input_stim3,[],destrect3);
        tmp_2nd_digit = floor(map_sbj.reward(visited_goal_state)-tmp_1st_digit*10);
        input_stim4 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_2nd_digit+1});
        sx4=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy4=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
        xpos4=xpos3 + sx4; ypos4=ypos2;
        destrect4=[xpos4-sx4/2,ypos4-sy4/2,xpos4+sx4/2,ypos4+sy4/2];
        Screen('DrawTexture', wPtr, input_stim4,[],destrect4);
        input_stim5 = Screen('MakeTexture', wPtr, img_set_num{1,11});
        sx5=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg*0.2);       sy5=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
        xpos5=xpos4 + sx5/2 + sx4/2; ypos5=ypos2;
        destrect5=[xpos5-sx5/2,ypos5-sy5/2,xpos5+sx5/2,ypos5+sy5/2];
        Screen('DrawTexture', wPtr, input_stim5,[],destrect5);
        tmp_3rd_digit = floor((map_sbj.reward(visited_goal_state)- 10*tmp_1st_digit-tmp_2nd_digit)*10);
        input_stim6 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_3rd_digit+1});
        sx6=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy6=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
        xpos6=xpos5 + sx5/2 + sx4/2; ypos6=ypos2;
        destrect6=[xpos6-sx6/2,ypos6-sy6/2,xpos6+sx6/2,ypos6+sy6/2];
        Screen('DrawTexture', wPtr, input_stim6,[],destrect6);
        
%         DrawFormattedText(wPtr, unicode_display,'center',ypos2+sy2+10); % add 'o' mark at the click pt.
%         DrawFormattedText(wPtr, [sprintf(' %.1f 동전을 획득했습니다.', map_sbj.reward(visited_goal_state))],'center',ypos2+sy2)
%         disp(sprintf('line:739, arrival=%d, reward=%d, condition=%d', original_rwd,state_sbj.reward_history(end),current_goal_state));
%        Screen(wPtr, 'Flip'); % display on
        if randperm(2,1) < 1.5
            if case_earn>0
                if outcome_state==9
                    current_sess_percent=current_sess_percent-(current_sess_percent)*(outcome_state-7)*rand()*(sqrt(105)-sqrt(trial))/sqrt(105)/4;
                elseif single_block_condition(4,trial)<1.5 % flexible and not 0 reached
                    current_sess_score=current_sess_score+9-outcome_state;
                    if outcome_state==6
                        current_sess_score=current_sess_score+1;
                    end
                    if outcome_state<7.5
                        current_sess_percent=current_sess_percent+(100-current_sess_percent)*(8-outcome_state)*rand()/4;
                    else
                        current_sess_percent=current_sess_percent-(current_sess_percent)*(outcome_state-7)*rand()/4;
                    end
                else % specific
                    current_sess_score=current_sess_score+4;
                    current_sess_percent=current_sess_percent+2*(100-current_sess_percent)*rand()/4;
                end
            else % not_earned
                current_sess_percent=current_sess_percent-2*(current_sess_percent)*rand()/4;
            end
            if randperm(2,1) < 1.5
                score_display = sprintf('%.1f',current_sess_score);
                unicode_display = [54788, 51116, 32, 49892, 54744, 32, 51216, 49688, 58, 32];
                for unicode_indx = 1:length(score_display)
                    if strcmp(score_display(unicode_indx),'.')
                        unicode_display = [unicode_display 46];
                    else
                        unicode_display = [unicode_display ceil(str2num(score_display(unicode_indx))+48)];
                    end
                end
                unicode_display = [unicode_display 32, 92, 110, 32, 92, 110, 32, 54788, 51116, 32, 49892, 54744, 32, 46321, 49688, 58, 32, 49345, 50948, 32];
                rank_display = sprintf('%.1f',100-current_sess_percent);
                for unicode_indx = 1:length(rank_display)
                    if strcmp(rank_display(unicode_indx),'.')
                        unicode_display = [unicode_display 46];
                    else
                        unicode_display = [unicode_display ceil(str2num(rank_display(unicode_indx))+48)];
                    end
                end
                unicode_display = [unicode_display 37];
                input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,6});
                %
                % (3) add outcome message
                xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
                sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
                sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
                xpos2=xpos; ypos2=destrect2(4)+sy2/2;%+50;%ypos;%+sy/2+sy2/2+50;
                destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
                % (4) display on
                Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
%                 DrawFormattedText(wPtr, [ sprintf('%.1f , %.1f',current_sess_score,100-current_sess_percent) '%'], 'center', destrect2(4));%,'','',1);                
                tmp_1st_digit = floor(current_sess_score/10);          
                tmp_2nd_digit = floor(current_sess_score-tmp_1st_digit*10);
                tmp_3rd_digit = floor((current_sess_score- 10*tmp_1st_digit-tmp_2nd_digit)*10);
                input_stim3 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_1st_digit+1});
                sx3=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy3=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos3=xpos - 4.2*sx3 - 1; ypos3=ypos2 + sy3;
                destrect3=[xpos3-sx3/2,ypos3-sy3/2,xpos3+sx3/2,ypos3+sy3/2];
                Screen('DrawTexture', wPtr, input_stim3,[],destrect3);
                input_stim4 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_2nd_digit+1});
                sx4=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy4=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos4=xpos3 + sx4; ypos4=ypos3;
                destrect4=[xpos4-sx4/2,ypos4-sy4/2,xpos4+sx4/2,ypos4+sy4/2];
                Screen('DrawTexture', wPtr, input_stim4,[],destrect4);
                input_stim5 = Screen('MakeTexture', wPtr, img_set_num{1,11});
                sx5=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg*0.2);       sy5=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos5=xpos4 + sx5/2 + sx4/2; ypos5=ypos4;
                destrect5=[xpos5-sx5/2,ypos5-sy5/2,xpos5+sx5/2,ypos5+sy5/2];
                Screen('DrawTexture', wPtr, input_stim5,[],destrect5);
                input_stim6 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_3rd_digit+1});
                sx6=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy6=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos6=xpos5 + sx5/2 + sx6/2; ypos6=ypos5;
                destrect6=[xpos6-sx6/2,ypos6-sy6/2,xpos6+sx6/2,ypos6+sy6/2];
                Screen('DrawTexture', wPtr, input_stim6,[],destrect6); 
                
                tmp_1st_digit = floor((100-current_sess_percent)/10);          
                tmp_2nd_digit = floor((100-current_sess_percent)-tmp_1st_digit*10);
                tmp_3rd_digit = floor(((100-current_sess_percent)- 10*tmp_1st_digit-tmp_2nd_digit)*10);
                input_stim3 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_1st_digit+1});
                sx3=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy3=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos3=xpos + sx3 ; ypos3=ypos2 + sy3;
                destrect3=[xpos3-sx3/2,ypos3-sy3/2,xpos3+sx3/2,ypos3+sy3/2];
                Screen('DrawTexture', wPtr, input_stim3,[],destrect3);
                input_stim4 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_2nd_digit+1});
                sx4=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy4=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos4=xpos3 + sx4; ypos4=ypos3;
                destrect4=[xpos4-sx4/2,ypos4-sy4/2,xpos4+sx4/2,ypos4+sy4/2];
                Screen('DrawTexture', wPtr, input_stim4,[],destrect4);
                input_stim5 = Screen('MakeTexture', wPtr, img_set_num{1,11});
                sx5=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg*0.2);       sy5=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos5=xpos4 + sx5/2 + sx4/2; ypos5=ypos4;
                destrect5=[xpos5-sx5/2,ypos5-sy5/2,xpos5+sx5/2,ypos5+sy5/2];
                Screen('DrawTexture', wPtr, input_stim5,[],destrect5);
                input_stim6 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_3rd_digit+1});
                sx6=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy6=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
                xpos6=xpos5 + sx5/2 + sx6/2; ypos6=ypos5;
                destrect6=[xpos6-sx6/2,ypos6-sy6/2,xpos6+sx6/2,ypos6+sy6/2];
                Screen('DrawTexture', wPtr, input_stim6,[],destrect6);    
%                 DrawFormattedText(wPtr, unicode_display,'center',ypos2+sy2+60); % add 'o' mark at the click pt.
    %             DrawFormattedText(wPtr, sprintf('Current Session Score: %d \n \nCurrent Session Ranking: Top %.1f%%', current_sess_score,100-current_sess_percent),'center',ypos2+sy2+20); % add 'o' mark at the click pt.
            end
        end        
        disp(sprintf('line:932, arrival=%d, reward=%d, condition=%d state= %d, %d actions=%d, %d', original_rwd,state_sbj.reward_history(end),current_goal_state, state_sbj.state_history(2), state_sbj.state_history(3),state_sbj.action_history(1),state_sbj.action_history(2)));
        Screen(wPtr, 'Flip'); % display on
        WaitSecs(0.5);        
        if(outcome_state~=9)
            value=outcome_state+0.1*case_earn;
        else
            value=outcome_state;
        end
        onset_state_from_trial_start=[onset_state_from_trial_start (GetSecs-trial_clock_start)];
        HIST_event_info0=[HIST_event_info0 [block; trial_in_block; state_sbj.index; ...
            (GetSecs-session_clock_start); (GetSecs-block_clock_start); onset_state_from_trial_start(end); value; current_goal_state]]; % event save

        if(strcmp(session_opt,'pre')==1) % pre-session
            WaitSecs(2.0);
        end
        if(strcmp(session_opt,'fmri')==1) % fmri-session
            WaitSecs(sec_reward_display);
        end

        %% Update HIST_event_info (overwrite at each block)
        HIST_event_info{1,index_num}=HIST_event_info0;
        
        %% update behavior matrix !@#$%
        if(isempty(HIST_behavior_info0))
            acc_rwd=0;
        else
            acc_rwd=HIST_behavior_info0(end,18);
        end
        mat_update=[block, trial_in_block, block_condition(1), block_condition(2), ...
            state_sbj.state_history(1), state_sbj.state_history(2), state_sbj.state_history(3), ...
            state_sbj.action_history(1), state_sbj.action_history(2), ...
            state_sbj.RT(1), state_sbj.RT(2), ...
            onset_state_from_trial_start(1), onset_state_from_trial_start(2), onset_state_from_trial_start(3), ...
            onset_action_from_trial_start(1), onset_action_from_trial_start(2), ...
            state_sbj.reward_history(end), acc_rwd+state_sbj.reward_history(end), ...
            current_goal_state, scenario((block-1)*trial_length+trial_in_block), seq(2,(block-1)*trial_length+trial_in_block), -1, current_sess_score, current_sess_percent];
        HIST_map_state_info0.map(tot_indx)=map_sbj_update;
        HIST_map_state_info0.state(tot_indx)=state_sbj_update;
        HIST_behavior_info0=[HIST_behavior_info0; mat_update];
        
           
        
        
    end % end of each trial

    %% confidence added 20190818
    conf=5;
    conf_ans_start_time=GetSecs;
    space=0;
    pktdata = [];
    while space==0&&(GetSecs-conf_ans_start_time)<10 % decision time llmit
        unicode_display = [50620, 47560, 45208, 32, 50612, 47140, 50912, 49845, 45768, 44620, 63, 32, 40, 48, 58, 32, 45320, 47924, 32, 50612, 47157, 45796, 32, 45, 32, 49, 48, 58, 32, 45320, 47924, 32, 49789, 45796, 41, 32];
        conf_display = sprintf('%d',conf);
        for unicode_indx = 1:length(conf_display)
            unicode_display = [unicode_display ceil(str2num(conf_display(unicode_indx))+48)];
        end
%         DrawFormattedText(wPtr, unicode_display, 'center', 'center');
%         DrawFormattedText(wPtr, ['How difficult(0~10)? ' sprintf('%d',conf)], 'center', 'center');
        input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,7});
        %
        % (3) add outcome message
        xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
        sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
        sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
        xpos2=xpos; ypos2=ypos;%+sy/2+sy2/2+50;
        destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
        % (4) display on
        Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
%         DrawFormattedText(wPtr, ['How difficult(0~10)? ' sprintf('%d',conf)], 'center', ypos2+sy2/2+10);%,'','',1);

        if conf < 10
            tmp_1st_digit = conf;
            input_stim3 = Screen('MakeTexture', wPtr, img_set_num{1,tmp_1st_digit+1});
            sx3=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy3=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
            xpos3=xpos; ypos3=ypos2 +sy2/2+ sy3/2;
            destrect3=[xpos3-sx3/2,ypos3-sy3/2,xpos3+sx3/2,ypos3+sy3/2];
            Screen('DrawTexture', wPtr, input_stim3,[],destrect3);
        else
            input_stim3 = Screen('MakeTexture', wPtr, img_set_num{1,2});
            input_stim4 = Screen('MakeTexture', wPtr, img_set_num{1,1});
            sx3=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy3=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
            xpos3=xpos - sx3/2 ; ypos3=ypos2 + sy2/2 + sy3/2;
            destrect3=[xpos3-sx3/2,ypos3-sy3/2,xpos3+sx3/2,ypos3+sy3/2];
            sx4=floor(NUM_MSG_SIZE(1)*disp_scale_goalimg);       sy4=floor(NUM_MSG_SIZE(2)*disp_scale_goalimg);
            xpos4=xpos + sx4/2 ; ypos4=ypos3;
            destrect4=[xpos4-sx4/2,ypos4-sy4/2,xpos4+sx4/2,ypos4+sy4/2];
            Screen('DrawTexture', wPtr, input_stim3,[],destrect3);
            Screen('DrawTexture', wPtr, input_stim4,[],destrect4);
        end

        Screen('Flip',wPtr);
        WaitSecs(0.1);
        [pktdata,treceived] = IOPort('read',myport,1,1)
        pktdata = ceil(pktdata);
        %[KeyIsDown, secs, keyCode] = KbCheck();
        if treceived>0 %(KeyIsDown)
            %[tmp ans_key_conf]=find(keyCode==1);
            ans_key_conf = pktdata;
        else
            ans_key_conf = 0;
        end
        if ismember(KEY_R,ans_key_conf) || ismember(KEY_C,ans_key_conf)
                if conf<10
                    conf=conf+1;
                end
        elseif ismember(KEY_L,ans_key_conf) || ismember(KEY_B,ans_key_conf)
                if conf>0
                    conf=conf-1;
                end
%         elseif ismember(KEY_B,ans_key_conf) %스페이스 바에서 r로 바꿈
%                 space=1;
%         elseif ismember(KEY_C,ans_key_conf) %스페이스 바에서 r로 바꿈
%                 space=1;
        end
    end    
    HIST_behavior_info0(end,22)=conf;
    % behavior matrix update
    HIST_map_state_info{1,index_num}=HIST_map_state_info0;
    HIST_behavior_info{1,index_num}=HIST_behavior_info0;
    disp(['line 1050:  [' num2str(HIST_behavior_info0(:).') ']']);
    disp(['line 1051:  [' num2str(HIST_event_info0(:).') ']']);
    disp(sprintf('line 1052:, arrival=%d, reward=%d, condition=%d state= %d, %d actions=%d, %d', original_rwd,state_sbj.reward_history(end),current_goal_state, state_sbj.state_history(2), state_sbj.state_history(3),state_sbj.action_history(1),state_sbj.action_history(2)));

    %% save the (updated) image usage matrix (overwriting)
    file_imgind_sv_name=[EXP_NAME '_info.mat'];
    file_name_sv=[pwd '\result_save\' file_imgind_sv_name];
    save(file_name_sv,'HIST_event_info','HIST_event_info_Tag','HIST_behavior_info','HIST_behavior_info_Tag','HIST_block_condition','HIST_block_condition_Tag');

end % end of each block


%% Ending message
str_end=sprintf('- Our experiments is over. Press any key to quit. -');
unicode_display = [49892, 54744, 32, 49464, 49496, 51060, 32, 45149, 45228, 49845, 45768, 45796, 46, 32, 50644, 53552, 47484, 32, 45580, 47084, 32, 49892, 54744, 51012, 32, 45149, 45236, 44256, 32, 49892, 54744, 51088, 51032, 32, 51648, 49884, 47484, 32, 44592, 45796, 47140, 51452, 49464, 50836, 46];
% DrawFormattedText(wPtr, unicode_display, 'center', 'center');
%DrawFormattedText(wPtr, str_end, 'center', 'center');
input_stim2 = Screen('MakeTexture', wPtr, img_set_msg{1,8});
%
% (3) add outcome message
xpos = round(screenWidth/2);    ypos = round(screenHeight/2);
sx=floor(IMAGE_SIZE(1)*disp_scale);       sy=floor(IMAGE_SIZE(2)*disp_scale);
sx2=floor(OUTCOME_MSG_SIZE(1)*disp_scale_goalimg);       sy2=floor(OUTCOME_MSG_SIZE(2)*disp_scale_goalimg);
xpos2=xpos; ypos2=ypos;%+sy/2+sy2/2+50;
destrect2=[xpos2-sx2/2,ypos2-sy2/2,xpos2+sx2/2,ypos2+sy2/2];
% (4) display on
Screen('DrawTexture', wPtr, input_stim2,[],destrect2);
Screen(wPtr, 'Flip');
KbWait; % temporarily disabled for test APR 21
% take a snapshot
if(DO_TAKE_SNAPSHOT==1)
    snapshot=Screen(wPtr, 'GetImage', [1, 1, floor(1.0*SCREEN_RESOLUTION(1)), floor(1.0*SCREEN_RESOLUTION(2))]);
    imageArray=[imageArray; {snapshot}];
end


%% save snapshots : CAUTION HEAVY PROCESS - might take a minute.
if(DO_TAKE_SNAPSHOT==1)
    for j=1:1:size(imageArray,1)
        str=sprintf('snapshot_dispay_exp_%03d.png',j);
        imwrite(imageArray{j},['snapshot\' str],'png');
    end
end


% behavior matrix update
HIST_behavior_info{1,index_num}=HIST_behavior_info0;


%% save the (updated) image usage matrix (overwriting)
file_imgind_sv_name=[EXP_NAME '_info.mat'];
file_name_sv=[pwd '\result_save\' file_imgind_sv_name];
save(file_name_sv,'HIST_event_info','HIST_event_info_Tag','HIST_behavior_info','HIST_behavior_info_Tag','HIST_block_condition','HIST_block_condition_Tag','HIST_map_state_info','HIST_map_state_info_Tag');

%% save all variables
file_sv_name=[EXP_NAME sprintf('_%d.mat',index_num)];
file_name_sv=[pwd '\result_save\' file_sv_name];
save(file_name_sv,'*');


%% session end sound
%%% SOUND DEACTIVATED BY SFW
% % % sec_dur_sound=1;
% % % Beeper('med', 0.4, sec_dur_sound); WaitSecs(sec_dur_sound)
% % % Beeper('high', 0.4, sec_dur_sound)



%% finish all
Screen('CloseAll');
clear mex
% clear Screen

disp('########################################################')
str_end1=sprintf('### session%d is done ############################',index_num);
disp(str_end1);
str_end2=sprintf('### next session = %d ############################',index_num+1);
disp(str_end2);
disp('########################################################')

% display the number of response failure
missed_count = length(find(HIST_event_info{1,index_num}(7,:)==-99));
disp(sprintf('- # of response failure in this session = %d. (will be penalized) ',missed_count));
diary off
movefile('diary.txt',diary_name);

IOPort('Close',myport)

end


