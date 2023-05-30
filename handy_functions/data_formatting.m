function data_formatting(session,folder,preint,postint)
addpath(genpath('/Users/kendranoneman/Projects/mayo/HelperFxns'))
types        =  {'pure','forward','backward'}; % types of trials 
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
data = load('-mat',sprintf('Users/kendranoneman/Projects/mayo/data/neural-decoding/raw/combinedMaestroSpkSortMTFEF.%s.mat',session));

[exp_clean,unitnames,snrs] = struct_clean(data.exp);

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

%trls = {exp_clean.dataMaestroPlx.trType}.';
%cellfun(@(q) str2double(q(9:11)), trls(:,2), 'uni', 0)  cellfun(@(q) str2double(q(15:16)), trls(:,2), 'uni', 0)
tt = [{exp_clean.dataMaestroPlx.trName}.' {exp_clean.dataMaestroPlx.trType}.' motionDirs stimContrasts motionSpeeds stimOnsets eyes_new];
trialTbl = cell2table(tt,'VariableNames',["TrialName","TrialType","Direction","Contrast","Speed","TargetMotionOnset","EyeTraces"]);
trialTbl.TrialName = categorical(string(trialTbl.TrialName)); trialTbl.TrialType = categorical(string(trialTbl.TrialType));


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
save(sprintf('%s/vars-%s-pre%03d-post%03d.mat',folder,session,preint,postint-800),'spike_times','pos','vels','acc','vels_times','-v7');
writetable(unitsTbl,sprintf('%s/units-%s-pre%03d-post%03d.csv',folder,session,preint,postint-800))

end