num_map=zeros(5,1); % max SPE / min SPE / max RPE / min RPE / random
fs_map=zeros(5,35); %0=flexible goal 1=specific
HL_map=zeros(5,35); %0=0.5/0.5 1=0.9/0.1
sc_map=zeros(4,141);
scenarios={'Max-SPE','Min-SPE','Max-RPE','Min-RPE','Random'};   
rnd_map=[]; % row = each cases, column: distances from each polices + avearage SPE
% for sbj=1:length(SBJ)
%     disp(sbj)
%     for sess=1:length(SBJ{1,sbj}.HIST_block_condition)
%         block=SBJ{1,sbj}.HIST_block_condition{1,sess};
%         for indx=1:length(block)
%             if block(5,indx)==1
%                 SBJ{1,sbj}.HIST_block_condition{1,sess}(5,indx)=SBJ{1,sbj}.HIST_block_condition{1,sess}(5,indx-1);
%             end
%         end
%     end
% end

for sbj=1:length(SBJ)
    disp(sbj)
    for sess=1:length(SBJ{1,sbj}.HIST_block_condition)
        block=SBJ{1,sbj}.HIST_block_condition{1,sess};
        for indx=1:3
            block_sc=block(3,35*indx);
            num_map(block_sc)=num_map(block_sc)+1;
            fs_map(block_sc,:)=(fs_map(block_sc,:)*(num_map(block_sc)-1)+(block(4,35*indx-34:35*indx))-1)/num_map(block_sc);
            HL_map(block_sc,:)=(HL_map(block_sc,:)*(num_map(block_sc)-1)+(block(5,35*indx-34:35*indx)>0.6))/num_map(block_sc);
            if block_sc==5
                temp=[block(4,35*indx-34:35*indx)-1, block(5,35*indx-34:35*indx)>0.5, SBJ{1,sbj}.regressor{1,1}.value(7,sort([105*indx-103:3:105*indx, 105*indx-102:3:105*indx])), mean(SBJ{1,sbj}.regressor{1,1}.value(7,sort([105*indx-103:3:105*indx, 105*indx-102:3:105*indx])))];
                rnd_map=[rnd_map;temp];
            else
                sc_map(block_sc,:)=[block(4,35*indx-34:35*indx)-1, block(5,35*indx-34:35*indx)>0.5, SBJ{1,sbj}.regressor{1,1}.value(7,sort([105*indx-103:3:105*indx, 105*indx-102:3:105*indx])), mean(SBJ{1,sbj}.regressor{1,1}.value(7,sort([105*indx-103:3:105*indx, 105*indx-102:3:105*indx])))];
            end
        end
    end
end
figure()
subplot(2,1,1)
hold on;
for ii=1:5
    plot(fs_map(ii,:))
end
legend(scenarios)
title('flex(1)-specific(2)')
subplot(2,1,2)
hold on;
for ii=1:5
    plot(HL_map(ii,:))
end
legend(scenarios)
title('0.9/0.1(0)-0.5/0.5(1)')

% figure()
% scatter(rnd_map(:,1),rnd_map(:,2),[],rnd_map(:,3),'filled')
% xlabel('% specific')
% ylable('% High uncertainty')

tot_rnd_num=length(rnd_map(:,141));
[~,rnd_indx]=sort(rnd_map(:,141));
figure()
subplot(2,1,1)
hold on;
for ii=tot_rnd_num:-1:tot_rnd_num*0.8
    plot(rnd_map(rnd_indx(ii),:))
end
subplot(2,1,2)
hold on;
for ii=1:tot_rnd_num*0.2
    plot(rnd_map(rnd_indx(ii),:))
end

figure()
hold on
plot(mean(rnd_map(rnd_indx([ceil(0.8*tot_rnd_num):end]),:),1))
plot(mean(rnd_map(rnd_indx([1:ceil(0.2*tot_rnd_num)]),:),1))
legend('high SPE', 'low SPE')