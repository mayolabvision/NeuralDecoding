function [spike_times,pos,contConditions] = data_formatting(session,folder,preint,postint)
addpath(genpath('/Users/kendranoneman/Projects/mayo/helperfunctions'))
% Purpose: takes in raw struct with recording data and converts it into a
% usable form for the Neural Decoding project

% Inputs:
% session  -->  name of session, as seen in .mat file name
%               e.g. session = 'pa29dir4A' from filename = combinedMaestroSpkSortMTFEF.pa29dir4A.mat
% folder   -->  location of raw .mat structure & where you want to store output
%               e.g. folder =  '/Users/kendranoneman/Projects/neural-decoding'
% preint   -->  how many time points before target motion onset to include (ms)
% postint  -->  how may time points after target motion onset to include (ms)
%               e.g. preint = 800; postint = 800;

% "spike_times": cell of size "number of neurons" x 1 
%                within spike_times{i} is a vector containing all the spike times of neuron i.
% "outputs":     matrix of size "number of recorded time points" x "number of features" that contains...
%                (1) x-position, (2) y-position, (3) x-velocity, (4) y-velocity 
% "output_times":   vector that states the time at all recorded time points 

%%%%%%%%%%%%%%%%% load session data %%%%%%%%%%%%%%%%%%%%
data = load('-mat',sprintf('/Users/kendranoneman/Data/neural-decoding/raw/combinedMaestroSpkSortMTFEF.%s.mat',session));

[exp_clean,unitnames,~] = struct_clean(data.exp);

% if you want to detect microsaccades
%[msFlag,eye_adjust] = cellfun(@(q,m) detect_msTrials(struct2cell(q),m,50,100,750,50), {exp_clean.dataMaestroPlx.mstEye}.', stimOnsets, 'uni', 0);
%exp_clean.dataMaestroPlx(logical(cell2mat(msFlag))) = []; eye_adjust = eye_adjust(~logical(cell2mat(msFlag))); 
%[exp_clean.dataMaestroPlx.mstEye] = eye_adjust{:};

% if you want to pull out particular conditions
%extract_conditions = {'d145','d235','d325','d415','c012','c100','sp10','sp20'};
%extract_columns = [2 3 4];
%define_columns = [2 3 4];
%[exp_clean,~] = struct_pullConditions(exp_clean,extract_conditions,extract_columns,define_columns);

tagS  =  {exp_clean.dataMaestroPlx.tagSection}.'; tagS = vertcat(tagS{:});
stimOnsets = {tagS.stTimeMS}.';

% if you want to detect pursuit onset time
%[pursuitOnsets,rxnTimes] = cellfun(@(q,m) detect_pursuitOnset(struct2cell(q),m,50,300), {exp_clean.dataMaestroPlx.mstEye}.', stimOnsets, 'uni', 0);
%exp_clean.dataMaestroPlx(isnan(cell2mat(rxnTimes))) = []; pursuitOnsets(isnan(cell2mat(rxnTimes))) = []; stimOnsets(isnan(cell2mat(rxnTimes))) = []; rxnTimes(isnan(cell2mat(rxnTimes))) = []; 
motionDirs = cellfun(@(q) str2double(q(strfind(q,'d')+1:strfind(q,'d')+3)), {exp_clean.dataMaestroPlx.trType}.', 'uni', 0);

% if you want to detect catch-up saccades
%[csTypes,ipt,saccProps] = cellfun(@(q,p,d) detect_catchupSaccade(struct2cell(q),p,d,1,200,750,30), {exp_clean.dataMaestroPlx.mstEye}.', pursuitOnsets, motionDirs, 'uni', 0);
%exp_clean.dataMaestroPlx(cell2mat(csTypes)==0) = []; pursuitOnsets(cell2mat(csTypes)==0) = []; stimOnsets(cell2mat(csTypes)==0) = []; rxnTimes(cell2mat(csTypes)==0) = []; motionDirs(cell2mat(csTypes)==0) = []; ipt(cell2mat(csTypes)==0) = []; saccProps(cell2mat(csTypes)==0) = []; csTypes(cell2mat(csTypes)==0) = [];
%new_condition = cellfun(@(x,y)[x,'_',types{y}], {exp_clean.dataMaestroPlx.condition_name}.',csTypes,'uni',0);
%[exp_clean.dataMaestroPlx.condition_name] = new_condition{:};

%exp_clean.dataMaestroPlx(cell2mat(csTypes)~=1) = []; pursuitOnsets(cell2mat(csTypes)~=1) = []; stimOnsets(cell2mat(csTypes)~=1) = []; rxnTimes(cell2mat(csTypes)~=1) = []; motionDirs(cell2mat(csTypes)~=1) = [];
eyes = {exp_clean.dataMaestroPlx.mstEye}.'; 
eyes_new = cellfun(@(et,so) trimSmooth_eyeTraces(et,so,preint,postint,20), eyes, stimOnsets, 'uni', 0); 

exp_clean.dataMaestroPlx(isnan(cellfun(@(q) q{1}(1), eyes_new,'uni', 1))) = []; 
motionDirs(isnan(cellfun(@(q) q{1}(1), eyes_new,'uni', 1))) = []; stimOnsets(isnan(cellfun(@(q) q{1}(1), eyes_new,'uni', 1))) = []; eyes_new(isnan(cellfun(@(q) q{1}(1), eyes_new,'uni', 1))) = [];

stimContrasts = cellfun(@(q) str2double(q(strfind(q,'c')+1:strfind(q,'c')+3)), {exp_clean.dataMaestroPlx.trType}.', 'uni', 0);
motionSpeeds = cellfun(@(q) str2double(q(strfind(q,'s')+2:strfind(q,'s')+3)), {exp_clean.dataMaestroPlx.trType}.', 'uni', 0);
%tt = [{exp_clean.dataMaestroPlx.trName}.' {exp_clean.dataMaestroPlx.trType}.' motionDirs pursuitOnsets stimOnsets rxnTimes eyes];
%trialTbl = cell2table(tt,'VariableNames',["TrialName","TrialType","Direction","PursuitOnset","TargetMotionOnset","RxnTime","EyeTraces"]);
%trialTbl.TrialName = categorical(string(trialTbl.TrialName)); trialTbl.TrialType = categorical(string(trialTbl.TrialType));

% eye acceleration variability
eyeAcc_std = cellfun(@(q) mean(std(q{3})), eyes_new, 'uni',0);
[~,idx_rank] = sort(cell2mat(eyeAcc_std));
idx_rank = [idx_rank (1:length(idx_rank)).'];

av_rank = zeros(size(eyeAcc_std));
av_rank(idx_rank(:,1)) = idx_rank(:,2);

%trls = {exp_clean.dataMaestroPlx.trType}.';
%cellfun(@(q) str2double(q(9:11)), trls(:,2), 'uni', 0)  cellfun(@(q) str2double(q(15:16)), trls(:,2), 'uni', 0)
tt = [{exp_clean.dataMaestroPlx.trName}.' {exp_clean.dataMaestroPlx.trType}.' motionDirs stimContrasts motionSpeeds stimOnsets eyeAcc_std num2cell(av_rank) eyes_new];
trialTbl = cell2table(tt,'VariableNames',["TrialName","TrialType","Direction","Contrast","Speed","TargetMotionOnset","EyeAcc_std","AV_rank","EyeTraces"]);
trialTbl.TrialNum = cellfun(@(q) str2double(q(end-3:end)), trialTbl.TrialName, 'uni', 1);
trialTbl = movevars(trialTbl,'TrialNum','After','TrialName');
trialTbl.TrialName = categorical(string(trialTbl.TrialName)); trialTbl.TrialType = categorical(string(trialTbl.TrialType));

%% testing that AV metric is sound
% av_10 = trialTbl.EyeAcc_std(trialTbl.Speed==10);
% av_20 = trialTbl.EyeAcc_std(trialTbl.Speed==20);
% 
% f1 = figure;
% t = tiledlayout(1,2);
% t.TileSpacing = 'loose';
% t.Padding = 'compact';
% 
% nexttile
% [datamean, datastd] = histStyle(av_10,'10 deg/s','AV','Number of Trials',[0 max(av_10)],[0 200],20,1,[0 0 0],[1 1 1]);
% nexttile
% [datamean, datastd] = histStyle(av_20,'20 deg/s','AV','Number of Trials',[0 max(av_20)],[0 200],20,1,[0 0 0],[1 1 1]);

lo_trl = 16;
hi_trl = 34;

f1 = figure;
t = tiledlayout(3,2);

x = (1:length(trialTbl.EyeTraces{1}{1}))-preint;

nexttile
plot(x,trialTbl.EyeTraces{lo_trl}{1}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{lo_trl}{1}(:,2),'r-')
ylabel('eye position (deg)')
title(sprintf('lo AV trial (trial %d, AV = %1.2f)',lo_trl,trialTbl.EyeAcc_std(lo_trl)))

nexttile
plot(x,trialTbl.EyeTraces{hi_trl}{1}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{hi_trl}{1}(:,2),'r-')
title(sprintf('hi AV trial (trial %d, AV = %1.2f)',hi_trl,trialTbl.EyeAcc_std(hi_trl)))

nexttile
plot(x,trialTbl.EyeTraces{lo_trl}{2}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{lo_trl}{2}(:,2),'r-')
ylabel('eye velocity (deg/s)')

nexttile
plot(x,trialTbl.EyeTraces{hi_trl}{2}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{hi_trl}{2}(:,2),'r-')

nexttile
plot(x,trialTbl.EyeTraces{lo_trl}{3}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{lo_trl}{3}(:,2),'r-')
ylabel('eye acceleration (deg/s^2)')

nexttile
plot(x,trialTbl.EyeTraces{hi_trl}{3}(:,1),'b-')
hold on
plot(x,trialTbl.EyeTraces{hi_trl}{3}(:,2),'r-')

xlabel(t,'time aligned to target motion onset (ms)')


%%
contConditions = cell(size(trialTbl,1),5);
for i=1:length(trialTbl.Contrast)
    contConditions{i,1} = repmat(trialTbl.TrialNum(i),preint+postint,1);
    contConditions{i,2} = repmat(trialTbl.Contrast(i),preint+postint,1);
    contConditions{i,3} = repmat(trialTbl.Speed(i),preint+postint,1);
    contConditions{i,4} = repmat(trialTbl.Direction(i),preint+postint,1);
    contConditions{i,5} = (1:preint+postint)';
end

contConditions = [vertcat(contConditions{:,1}) vertcat(contConditions{:,2}) vertcat(contConditions{:,3}) vertcat(contConditions{:,4}) vertcat(contConditions{:,5})];

condition_strings = cellstr(num2str(contConditions(:,2:4)));
unique_conditions = unique(condition_strings);
condition_map = containers.Map(unique_conditions, 1:numel(unique_conditions));

contConditions(:, 6) = cell2mat(values(condition_map, condition_strings));

% Directions in this session (4, corrected by rotation factor)
dirsdeg  =  sort(unique(trialTbl.Direction)); % direction for each trial 
contrasts  =  sort(unique(trialTbl.Contrast)); % direction for each trial
speeds  =  sort(unique(trialTbl.Speed)); % direction for each trial 

%%%%%%%%%%%%%%% Unit Info %%%%%%%%%%%%%%%%%%%
ut = makeUnitsTable_fromStruct(exp_clean,trialTbl,250);
if isequal(session(2),'a')
    monk = 'aristotle';
elseif isequal(session(2),'b')
    monk = 'batman';
end
ut = [cellstr(repmat(categorical(string(monk)),length(unitnames),1)) cellstr(repmat(categorical(string(session)),length(unitnames),1)) ut];
varNames   =  ["Monkey","Session","UnitName","Sess_Unit","BrainArea","SNR","BestDir","NullDir","PrefDirFit","mnFR_bestDir","varFR_bestDir","DepthMod","SelDir","DI","SI","signiffl","SpikeTimes"];
unitsTbl   =  cell2table(ut,"VariableNames",varNames); 
unitsTbl.Monkey = categorical(string(unitsTbl.Monkey)); unitsTbl.Session = categorical(string(unitsTbl.Session));
unitsTbl.UnitName = categorical(string(unitsTbl.UnitName)); unitsTbl.Sess_Unit = categorical(string(unitsTbl.Sess_Unit)); unitsTbl.BrainArea = categorical(string(unitsTbl.BrainArea));
unitsTbl.SpikeTimes = [];

units  =   {exp_clean.dataMaestroPlx.units}.'; 
for t = 1:size(trialTbl,1) % for each trial
    shift       =  trialTbl.TargetMotionOnset(t);    % time stimulus starts to move

    for u = 1:length(unitnames) % for each unit
        thisunit  =  unitnames{u}; % unit name

        if isfield(units{t}, (thisunit)) % if unit fired at some point in the trial
            spktimes     =  units{t,1}.(thisunit)-shift; % shift spike times so aligned to 'onset'
            spkind       =  spktimes >= -preint & spktimes < postint; % only include spikes in time window
            alignedspks  =  spktimes(spkind);

            % Calculate firing rate in Hz (spks/sec)
            if ~isempty(alignedspks) % if unit fired during specific time window
                spike_times{u}{t} = (alignedspks + preint) + ((t-1)*(preint+postint));
            end
        end
    end
end

spike_times = cellfun(@(z) vertcat(z{:}), spike_times, 'uni', 0);

pos = cellfun(@(q) q{1}, eyes_new, 'uni', 0); pos = vertcat(pos{:});
vels = cellfun(@(q) q{2}, eyes_new, 'uni', 0); vels = vertcat(vels{:});
acc = cellfun(@(q) q{3}, eyes_new, 'uni', 0); acc = vertcat(acc{:});

vels_times     =  (1:size(trialTbl,1)*(preint+postint))';

%%%%%%%%%%%%%%% Save to a file %%%%%%%%%%%%%%%%%%
save(sprintf('%s/vars/vars-%s-pre%03d-post%03d.mat',folder,session,preint,postint-800),'spike_times','pos','vels','acc','vels_times','contConditions','-v7');
writetable(unitsTbl,sprintf('%s/units/units-%s-pre%03d-post%03d.csv',folder,session,preint,postint-800))

end
