clear
clc

tic
%% Setup folders and parameters

folder = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets';
dataFolder = '/Users/kendranoneman/Data/neural-decoding/raw';

addpath(genpath('/Users/kendranoneman/Projects/mayo/helperfunctions'))

filelist   =  dir(dataFolder); % files in raw data folder
name       =  {filelist.name}; 
sessions   =  name(~strncmp(name, '.', 1))';
%sessions = sessions(16);

%%
parpool('local',8);
warning('off')
parfor s=1:length(sessions) % loop through each session
    sess_name  =  string(sessions{s}); sess_name = char(extractBetween(sess_name,".","."));
    fprintf(sprintf('%d / %d \n\n',s, length(sessions)));

%     %%%%%%%%%%%%%%% Save to a file %%%%%%%%%%%%%%%%%%
%     if not(isfolder(sprintf('%s/%s',folder,sess_name)))
%         mkdir(sprintf('%s/%s',folder,sess_name))
%     end

    [spike_times,pos,contConditions] = data_formatting(sess_name,folder,500,300+800);
end

toc
load gong.mat
sound(y)
