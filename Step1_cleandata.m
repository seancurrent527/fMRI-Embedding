% Anita Shankar shankar.85@osu.edu 
% last updated 12/03/2020
% This script takes connectivity matrices and behavioral data and cleans
% and organizes it for further use in ML applications


%% initialize variables and filepaths
clear
% read in dataset info & demographics
demographics=readtable('ndar_subject01.txt');
% get rid of irrelevant information
demographics(:,[1:4,6,16:end])=[];
% sort table by subject number (lowest to highest)
demographics=sortrows(demographics,1);
% identify the missing subjects and remove them
% create data structures with sublist, age
sublist=demographics.src_subject_id;
age=demographics.interview_age/12;
clear demographics missing_idx



 %% find older adults
idx=[];
idx=find(age>65);
age=age(idx);
sublist=sublist(idx);

%% remove subs with bad data (using visual inspection)
% any missing nodes
%badsubs={'HCA6472071','HCA6595794','HCA7027863','HCA7468388','HCA7913181','HCA8456386','HCA9170474','HCA9285794','HCA6691992'};
% worst subs
badsubs={'HCA6472071','HCA7027863','HCA7913181','HCA7468388','HCA8456386','HCA9170474','HCA6691992'}; %

badsubidx=[];
for ii=1:length(badsubs)
    badsubidx(ii,1)=find(strcmp(sublist,badsubs(ii))==1);
end

age(badsubidx)=[];
sublist(badsubidx)=[];
%% remove subs with high motion
meanFDtable=readtable('MeanFD_VISMOTOR.csv');
meanFDname=erase(meanFDtable.Var1,'sub-');
temp_idx=[];
for ii=1:length(sublist) %for all the subjects
    temp_idx(ii,1)=find(strcmp(sublist(ii),meanFDname)==1);
end
meanFD=meanFDtable.Var2(temp_idx);



%motionsub_idx=find(meanFD(:>.2);
%% load behavioral task perfomance data

myDir = '/Users/anita/Documents/MATLAB/Data/HCP_MAPS/BEHAV';

for z=1:length(sublist) %for all the subjects
    temp = readtable(strcat(myDir,'/VISMOTOR_',sublist{z},'_behav.csv')); %read in the subject file
    vismotor.trialNum(1:27,z)=temp.trialNum(5:end);
    vismotor.rt(1:27,z)=temp.firstRt(5:end);
    vismotor.correct(1:27,z)=temp.correct(5:end);
    vismotor.side(1:27,z)=temp.side(5:end);
    correct_idx{1,z}=num2cell(find(string(vismotor.correct(:,z))=='True'));
    acc(1,z)=length(correct_idx{1,z})/27;
end

 vismotor.correct_idx=correct_idx;

% mean_rxn time
 for z=1:length(sublist) %for all the subjects
     idx=cell2mat(correct_idx{1,z});
     rxn=vismotor.rt(idx,z);
     mean_rxn(1,z)=nanmean(rxn);
 end
clear temp z myDir correct_idx acc correct_idx

mean_rxn=mean_rxn';
clear vismotor idx p r rxn 
%% plotting
figure(1)
scatter(age,mean_rxn,'filled')
lsline

[r,p]=corr(age,mean_rxn);
fprintf('correlation age with mean reaction time: R:%.4f, p=%.4f \n',r,p); 

[r,p]=corr(mean_rxn,meanFD);
fprintf('correlation meanFD with mean reaction time: R:%.4f, p=%.4f \n',r,p);


[r,p]=corr(age,meanFD);
fprintf('correlation age with mean FD: R:%.4f, p=%.4f \n',r,p); 
 
%% Load in MRI data
num_nodes=268; %number of nodes in the used atlas
num_subs=length(sublist);
load('roi_labels.mat');
myDir = '/Users/anita/Documents/MATLAB/Data/HCP_MAPS/MRI';

%%
temp=nan(num_nodes,num_nodes,num_subs); %create a temp matrix full of Nans
for z=1:length(sublist) %for all the subjects
    mat = csvread(strcat(myDir,'/_sub-',sublist{z},'_ses-01_task-VISMOTOR_run-1_shen2func_correlation_connMatdf.csv')); % upload the person's data as 'mat'
        for i=2:size(mat,1) %for all the rows in mat starting at row 2 (which is where the data starts)
            x=mat(i,1); % x is the data by rows
                for j=2:size(mat,2) %for all the columns in mat
                    y=mat(1,j); %y is the columns
                        temp(x,y,z)=mat(i,j);
                end
        end
end

full_conmat=temp;

%%
clear i j mat myDir num_nodes num_subs p r temp x y z

