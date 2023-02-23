
% "spike_times": cell of size "number of neurons" x 1 
%                within spike_times{i} is a vector containing all the spike times of neuron i.
% "outputs":     matrix of size "number of recorded time points" x "number of features" that contains...
%                (1) x-position, (2) y-position, (3) x-velocity, (4) y-velocity 
% "output_times":   vector that states the time at all recorded time points 

%%%%%%%%%%%%%%%%% Load session data %%%%%%%%%%%%%%%%%%%%
clear 
clc

session  =  'pa29dir4A';
preint   =  800;
postint  =  800;

folder  =  '/Users/kendranoneman/Projects/mayo/data/neural-decoding';
data = load('-mat',sprintf('%s/raw/combinedMaestroSpkSortMTFEF.%s.mat',folder,session));

data.exp.dataMaestroPlx(find(cellfun(@isempty, {data.exp.dataMaestroPlx.units}.'))) = []; % throw out empty trials

%
%%%%%%%%%%%%%%%% Trial Info %%%%%%%%%%%%%%%%%%%%
rotFactor  =  -1*double(data.exp.info.rotfactor); % rotation factor

trialVars = ["TrialName","TrialType","Direction","Contrast","Speed","TrialLength","TargetMotionOnset","PursuitOnset","RxnTime","msFlag","Eye_Traces","Target_Traces","Times"];
trialTbl = cell([length(data.exp.dataMaestroPlx), length(trialVars)]);

for t=1:size(data.exp.dataMaestroPlx,2) % loop through each trial
    trial_name    =  data.exp.dataMaestroPlx(t).trName;
    trial_type    =  data.exp.dataMaestroPlx(t).trType;
    direction     =  str2double(data.exp.dataMaestroPlx(t).trType(4:6)) - rotFactor; 
    contrast      =  str2double(data.exp.dataMaestroPlx(t).trType(9:11));  
    speed         =  str2double(data.exp.dataMaestroPlx(t).trType(15:16)); 
    trial_length  =  data.exp.dataMaestroPlx(t).trial.mstDur; 
    motion_onset  =  data.exp.dataMaestroPlx(t).tagSection.stTimeMS; 
    
    if ~isempty(motion_onset)
        eye     =  cellfun(@(x) x(1:end),struct2cell(data.exp.dataMaestroPlx(t).mstEye),'uni',0);
        target  =  num2cell([data.exp.dataMaestroPlx(t).target.pos; data.exp.dataMaestroPlx(t).target.vel],2);
    
        % Calculate radial pos/vel, pursuit onset, detect micro- and catchup- saccades
        [pursuit_onset,rxn_time,eye_traces,target_traces,msFlag] = detect_eyetraces(eye,target,motion_onset,preint,postint);
     
        if msFlag==0 && ~isnan(rxn_time) % no microsaccades
            trialTbl(t,:)  =  {categorical(string(trial_name)), categorical(string(trial_type)), direction, contrast, speed, trial_length, motion_onset, pursuit_onset, rxn_time, msFlag, eye_traces, target_traces, {((1:(preint+postint)))'}};     
        end
    end
end

% Only use trials with detected pursuit onset, no microsaccades, and no motion pulses during fixation
toss                =  ~cellfun(@isempty, trialTbl); gTrials = toss(:,1); 
trialTbl            =  cell2table(trialTbl(gTrials,:),"VariableNames",trialVars);
cumTime             =  [1; (cumsum(trialTbl.TrialLength + 1))];
trialTbl.StartTime  =  cumTime(1:end-1);

trName        =  {data.exp.dataMaestroPlx.trName}.'; % names of "good" trials
data.exp.dataMaestroPlx(~cellfun(@(y) ismember(y,trialTbl.TrialName),trName)) = []; % throw out "bad" trials from raw data structure

% Directions (4, corrected by rotation factor)
dirsdeg  =  sort(unique(trialTbl.Direction)); % direction for each trial

trialTbl.Number = (1:size(trialTbl,1))';

outputs  =  trialTbl.Eye_Traces;
outputs  =  [vertcat(outputs{:,1}) vertcat(outputs{:,2}) vertcat(outputs{:,3}) vertcat(outputs{:,4}) vertcat(outputs{:,5}) vertcat(outputs{:,6}) vertcat(outputs{:,7}) vertcat(outputs{:,8})];

pos  =  outputs(:,1:2);
vels  =  outputs(:,3:4);

output_times = rowfun(@add_time, trialTbl, 'InputVariables',{'Times','StartTime'},'OutputFormat','cell');
vel_times = vertcat(output_times{:});

%%
%%%%%%%%%%%%%%% Unit Info %%%%%%%%%%%%%%%%%%%
unitVars  = ["UnitName","BrainArea","SNR","BestDir","MeanFR_BestDir","VarFR_BestDir","MeanFR_perDir","VarFR_perDir"];

% Monkey name
if isequal(session(2),'a')
    monk  =  'aristotle';
elseif isequal(session(2),'b')
    monk  =  'batman';
end

% Channels and SNRs
channels =  data.exp.info.channels;  % names of all channels
snrs     =  data.exp.info.SNRs;      % SNR for each channel

all_units = cellfun(@(x) fieldnames(x), {data.exp.dataMaestroPlx.units}.', 'uni', 0);
[B,BG] = groupcounts(vertcat(all_units{:}));
[C,ia] = setdiff(channels,cellfun(@(y) y(end-3:end), BG(B==max(B)), 'uni', 0));

channels(ia) = []; snrs(ia) = [];
[unitnames,I] = sort(channels); snrs = snrs(I);

unitnames  =  cellfun(@(z) strcat('unit',z), unitnames, 'uni', 0)';

% If >24 then MT, <24 then FEF
unitNum     =  cellfun(@(y) sscanf(y,'unit%d')>24, unitnames,'uni',1);
brainareas  =  cell(length(unitNum),1);
brainareas(unitNum==1,1)  =  {'MT'};
brainareas(unitNum==0,1)  =  {'FEF'};

 %%%%%%%%%%%%%%%%%%%%%% local motion window for pure pursuit trials %%%%%%%%%%%%%%%%%%%%%%
[spkcnts,spike_times] = deal(cell(length(unitnames),1));
badtrl_names = cell(length(dirsdeg),1);
for d=1:length(dirsdeg) % for each pursuit direction
    %%%%%%%%%%%%%%%% pure trials %%%%%%%%%%%%%%%%5
    trls   =   trialTbl(trialTbl.Direction==dirsdeg(d),:); 
    units  =   {data.exp.dataMaestroPlx(trialTbl.Direction==dirsdeg(d)).units}.'; 

    for t = 1:size(trls,1) % for each trial
        time_range  =  [trls.TargetMotionOnset(t)-preint trls.TargetMotionOnset(t)+postint];
        spkwin      =  [trls.TargetMotionOnset(t) trls.TargetMotionOnset(t)+250];
        start_time  =  trls.StartTime(t);

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

spike_times = cellfun(@(z) vertcat(z{:}), spike_times, 'uni', 0);

mnFRByDir = cellfun(@(q) cellfun(@nanmean, q), spkcnts, 'uni', 0);
varFRByDir = cellfun(@(q) cellfun(@nanvar, q), spkcnts, 'uni', 0);

[mnFRbestdir,bestdir] = max(cell2mat(mnFRByDir),[],2);
bestDir = dirsdeg(bestdir);
varFRbestdir = cell2mat(varFRByDir); varFRbestdir = varFRbestdir(bestdir);

unitsTbl  =  [cellstr(categorical(string(unitnames))),cellstr(categorical(string(brainareas))),num2cell(snrs'),num2cell(bestDir,2),num2cell(mnFRbestdir,2),num2cell(varFRbestdir,2),mnFRByDir,varFRByDir];       
unitsTbl  =  cell2table(unitsTbl,'VariableNames',unitVars);
unitsTbl.UnitName   =  categorical(string(unitsTbl.UnitName));
unitsTbl.BrainArea  =  categorical(string(unitsTbl.BrainArea));

% Save to a file
save(sprintf('%s/preprocessed/MTFEF-%s-1600ms.mat',folder,session),'spike_times','pos','vels','vel_times');

function eye_times = add_time(tms,a)
    eye_times = tms{1} + (a-1);
end