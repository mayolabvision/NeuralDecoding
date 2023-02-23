function data_formatting(session,folder,preint,postint)
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
data = load('-mat',sprintf('%s/raw/combinedMaestroSpkSortMTFEF.%s.mat',folder,session));
data.exp.dataMaestroPlx(find(cellfun(@isempty, {data.exp.dataMaestroPlx.units}.'))) = []; % throw out empty trials

% the directions of pursuit for each session are rotated by some factor
% e.g. if rotFactor = -10, then directions = [10 100 190 280] degrees
rotFactor  =  -1*double(data.exp.info.rotfactor); % same w/in session

%%%%%%%%%%%%%%%%%% characterize individual trials %%%%%%%%%%%%%%%%%%%%
% initialize a cell array to store calculated parameters in
trialVars = ["TrialName","TrialType","Direction","Contrast","Speed","TrialLength","TargetMotionOnset","PursuitOnset","RxnTime","msFlag","Eye_Traces","Target_Traces","Times"];
trialTbl = cell([length(data.exp.dataMaestroPlx), length(trialVars)]);

% Loop through each individual trial to pull out features
for t = 1:size(data.exp.dataMaestroPlx,2) % for each trial
    trial_name    =  data.exp.dataMaestroPlx(t).trName;               % trial name
    trial_type    =  data.exp.dataMaestroPlx(t).trType;               % trial condition
    direction     =  str2double(trial_type(4:6))-rotFactor;           % motion direction (in degrees)
    contrast      =  str2double(trial_type(9:11));                    % stimulus contrast (100% = full brightness)
    speed         =  str2double(trial_type(15:16));                   % motion speed (in deg/s)
    trial_length  =  data.exp.dataMaestroPlx(t).trial.mstDur;         % length of trial (ms)
    motion_onset  =  data.exp.dataMaestroPlx(t).tagSection.stTimeMS;  % time stimulus starts to move (ms)
    
    if ~isempty(motion_onset) % make sure the trial has info about the stimulus
        % horizontal position, vertical position, horizontal velocity, vertical velocity
        eye     =  cellfun(@(x) x(1:end),struct2cell(data.exp.dataMaestroPlx(t).mstEye),'uni',0);
        target  =  num2cell([data.exp.dataMaestroPlx(t).target.pos; data.exp.dataMaestroPlx(t).target.vel],2);
    
        % calculate radial pos/vel, pursuit onset, detect microsaccades
        [pursuit_onset,rxn_time,eye_traces,target_traces,msFlag]  =  detect_eyetraces(eye,target,motion_onset,preint,postint);
     
        % don't include trials w/ microsaccades or no pursuit onset 
        if msFlag == 0 && ~isnan(rxn_time) 
            trialTbl(t,:)  =  {categorical(string(trial_name)), categorical(string(trial_type)), direction, contrast, speed, trial_length, motion_onset, pursuit_onset, rxn_time, msFlag, eye_traces, target_traces, {((1:(preint+postint)))'}};     
        end
    end
end % end trials

% Remove the trials you didn't include above from the original data file
toss          =  ~cellfun(@isempty, trialTbl); gTrials = toss(:,1);          % which trials to remove?
trialTbl      =  cell2table(trialTbl(gTrials,:),"VariableNames",trialVars);  % convert cell array to table w/ headings
trName        =  {data.exp.dataMaestroPlx.trName}.';                         % names of all trials
data.exp.dataMaestroPlx(~cellfun(@(y) ismember(y,trialTbl.TrialName),trName)) = []; 

% To format the eye traces into a continuous stream, calculate the time in which each trial starts
cumTime             =  [1; (cumsum(trialTbl.TrialLength + 1))];                  
trialTbl.StartTime  =  cumTime(1:end-1);
trialTbl.Number = (1:size(trialTbl,1))';

% Directions in this session (4, corrected by rotation factor)
dirsdeg  =  sort(unique(trialTbl.Direction)); % direction for each trial

%%%%%%%%%%%%%%% Unit Info %%%%%%%%%%%%%%%%%%%
% initialize a cell array to store calculated parameters in
unitVars  = ["UnitName","BrainArea","SNR","BestDir","MeanFR_BestDir","VarFR_BestDir","MeanFR_perDir","VarFR_perDir"];

% Channels and SNRs
channels  =  data.exp.info.channels;  % names of all channels
snrs      =  data.exp.info.SNRs;      % SNR for each channel

% Only include units that fire at least once during every trial
all_units  =  cellfun(@(x) fieldnames(x), {data.exp.dataMaestroPlx.units}.', 'uni', 0);
[B,BG]     =  groupcounts(vertcat(all_units{:}));
[~,ia]     =  setdiff(channels,cellfun(@(y) y(end-3:end), BG(B==max(B)), 'uni', 0));
channels(ia)   =  [];              snrs(ia)  =  [];
[unitnames,I]  =  sort(channels);  snrs      =  snrs(I);

% Names of the units recorded in this session
unitnames  =  cellfun(@(z) strcat('unit',z), unitnames, 'uni', 0)';

% What brain area were these units from? 
% If > 24 then MT, <= 24 then FEF
unitNum     =  cellfun(@(y) sscanf(y,'unit%d')>24, unitnames,'uni',1);
brainareas  =  cell(length(unitNum),1);
brainareas(unitNum==1,1)  =  {'MT'};
brainareas(unitNum==0,1)  =  {'FEF'};

[spkcnts,spike_times]  =  deal(cell(length(unitnames),1));

% Loop through each motion direction, trial, and unit
for d = 1:length(dirsdeg) % for each direction

    trls   =   trialTbl(trialTbl.Direction==dirsdeg(d),:); 
    units  =   {data.exp.dataMaestroPlx(trialTbl.Direction==dirsdeg(d)).units}.'; 

    for t = 1:size(trls,1) % for each trial
        shift       =  trls.TargetMotionOnset(t);    % time stimulus starts to move
        time_range  =  [shift-preint shift+postint]; % time window to pull out spikes (before & after shift)
        spkwin      =  [shift shift+250];            % time window to calculate best direction 
        start_time  =  trls.StartTime(t);            % time individual trial starts 

        for u = 1:length(unitnames) % for each unit
            thisunit  =  unitnames{u}; % unit name

            if isfield(units{t}, (thisunit)) % if unit fired at some point in the trial
                spktimes     =  units{t,1}.(thisunit); 
                spkind       =  spktimes >= time_range(1) & spktimes < time_range(2); % only include spikes in time window
                alignedspks  =  spktimes(spkind);

                % Calculate firing rate in Hz (spks/sec)
                if ~isempty(alignedspks) % if unit fired during specific time window
                    spike_times{u}{trls.Number(t)} = alignedspks + (start_time-1);
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


unitsTbl  =  [cellstr(categorical(string(unitnames))),cellstr(categorical(string(brainareas))),num2cell(snrs'),num2cell(bestDir,2),num2cell(mnFRbestdir,2),num2cell(varFRbestdir,2),mnFRByDir,varFRByDir];       
unitsTbl  =  cell2table(unitsTbl,'VariableNames',unitVars);
unitsTbl.UnitName   =  categorical(string(unitsTbl.UnitName)); unitsTbl.BrainArea  =  categorical(string(unitsTbl.BrainArea));

%%%%%%%%%%%%%%% Save to a file %%%%%%%%%%%%%%%%%%
% size = # of neurons x 1
spike_times = cellfun(@(z) vertcat(z{:}), spike_times, 'uni', 0);

outputs  =  trialTbl.Eye_Traces;
outputs  =  [vertcat(outputs{:,1}) vertcat(outputs{:,2}) vertcat(outputs{:,3}) vertcat(outputs{:,4}) vertcat(outputs{:,5}) vertcat(outputs{:,6}) vertcat(outputs{:,7}) vertcat(outputs{:,8})];

pos           =  outputs(:,1:2); vels  =  outputs(:,3:4);
output_times  =  rowfun(@add_time, trialTbl, 'InputVariables',{'Times','StartTime'},'OutputFormat','cell');
vel_times     =  vertcat(output_times{:});

save(sprintf('%s/preprocessed/MTFEF-%s-1600ms.mat',folder,session),'spike_times','pos','vels','vel_times','-v7');

function eye_times = add_time(tms,a)
    % for adding trial start time to each row in table
    eye_times = tms{1} + (a-1);
end
end