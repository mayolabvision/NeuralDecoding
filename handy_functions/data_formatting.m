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

%tt = [{exp_clean.dataMaestroPlx.trName}.' {exp_clean.dataMaestroPlx.trType}.' motionDirs pursuitOnsets stimOnsets rxnTimes eyes];
%trialTbl = cell2table(tt,'VariableNames',["TrialName","TrialType","Direction","PursuitOnset","TargetMotionOnset","RxnTime","EyeTraces"]);
%trialTbl.TrialName = categorical(string(trialTbl.TrialName)); trialTbl.TrialType = categorical(string(trialTbl.TrialType));

tt = [{exp_clean.dataMaestroPlx.trName}.' {exp_clean.dataMaestroPlx.trType}.' motionDirs stimOnsets eyes_new];
trialTbl = cell2table(tt,'VariableNames',["TrialName","TrialType","Direction","TargetMotionOnset","EyeTraces"]);
trialTbl.TrialName = categorical(string(trialTbl.TrialName)); trialTbl.TrialType = categorical(string(trialTbl.TrialType));

% Directions in this session (4, corrected by rotation factor)
dirsdeg  =  sort(unique(trialTbl.Direction)); % direction for each trial 

%%%%%%%%%%%%%%% Unit Info %%%%%%%%%%%%%%%%%%%
% if you want to find each neurons' preferred/anti-preferred direction
% initialize a cell array to store calculated parameters in
unitVars  = ["UnitName","BrainArea","SNR","BestDir","MeanFR_BestDir","VarFR_BestDir"];

% What brain area were these units from? 
% If > 24 then MT, <= 24 then FEF
unitNum     =  cellfun(@(y) sscanf(y,'unit%d')>24, unitnames,'uni',1);
brainareas  =  cell(length(unitNum),1);
brainareas(unitNum==1,1)  =  {'MT'};
brainareas(unitNum==0,1)  =  {'FEF'};

[spkcnts,spike_times]  =  deal(cell(length(unitnames),1));
spkwin      =  [0 250];           

% Loop through each motion direction, trial, and unit
for d = 1:length(dirsdeg) % for each direction
    trls   =   trialTbl(trialTbl.Direction==dirsdeg(d),:); 
    units  =   {exp_clean.dataMaestroPlx(trialTbl.Direction==dirsdeg(d)).units}.'; 
    for t = 1:size(trls,1) % for each trial
        shift       =  trls.TargetMotionOnset(t);    % time stimulus starts to move
        for u = 1:length(unitnames) % for each unit
            thisunit  =  unitnames{u}; % unit name

            if isfield(units{t}, (thisunit)) % if unit fired at some point in the trial
                spktimes     =  units{t,1}.(thisunit)-shift; % shift spike times so aligned to 'onset'
                spkind       =  spktimes >= -preint & spktimes < postint; % only include spikes in time window
                alignedspks  =  spktimes(spkind);

                % Calculate firing rate in Hz (spks/sec)
                if ~isempty(alignedspks) % if unit fired during specific time window
                    spksHz  =  (sum(alignedspks>=spkwin(1) & alignedspks<spkwin(2))/(abs(spkwin(2)-spkwin(1))))*1000;
                else % unit did not spike in time window
                    spksHz  =  0;
                end

            else % no information about unit in this trial at all
                spksHz  =  NaN; 
            end
            % {unit}{direction}(trial)
            spkcnts{u}{d}(t) = spksHz;
        end
    end
end

% For each unit, determine its "best direction" from the trial-averaged firing rates
mnFRByDir   =  cellfun(@(q) cellfun(@nanmean, q), spkcnts, 'uni', 0);  % mean FR in each direction
varFRByDir  =  cellfun(@(q) cellfun(@nanvar, q), spkcnts, 'uni', 0);   % var FR in each direction

% best direction = direction in which individual neuron fired on average the most
[mnFRbestdir,bestdir]  =  max(cell2mat(mnFRByDir),[],2);  % mean FR in best direction 
bestDir                =  dirsdeg(bestdir);               % best direction
varFRbestdir           =  cell2mat(varFRByDir);           
varFRbestdir           =  varFRbestdir(bestdir);          % var FR in best direction

unitsTbl  =  [cellstr(categorical(string(unitnames))),cellstr(categorical(string(brainareas))),num2cell(snrs'),num2cell(bestDir,2),num2cell(mnFRbestdir,2),num2cell(varFRbestdir,2)];       
unitsTbl  =  cell2table(unitsTbl,'VariableNames',unitVars);
unitsTbl.UnitName   =  categorical(string(unitsTbl.UnitName)); unitsTbl.BrainArea  =  categorical(string(unitsTbl.BrainArea));
    
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