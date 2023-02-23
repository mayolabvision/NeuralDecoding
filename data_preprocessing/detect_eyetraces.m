function [pursuit_onset,rxn_time,eye_traces,target_traces,msFlag] = detect_eyetraces(eye,target,motion_onset,preint,postint)

% HEPos, VEPos, HEVel, VEVel
% add polar coordinates
[thPeye,rhoPeye] = cart2pol(eye{1},eye{2});
[thVeye,rhoVeye] = cart2pol(eye{3},eye{4});
eye_traces = [eye; num2cell([rhoPeye;thPeye;rhoVeye;thVeye],2)];

[thPos,rhoPos] = cart2pol(target{1},target{2});
[thVel,rhoVel] = cart2pol(target{3},target{4});
target_traces = [target; num2cell([rhoPos;thPos;rhoVel;thVel],2)];

% detect pursuit onset
[~,rhoVeye] = cart2pol(smoothdata(eye{3},'gaussian',20),smoothdata(eye{4},'gaussian',20)); 
vBase = rhoVeye(motion_onset-50:motion_onset+50);
baseVel = mean(vBase); % baseline eye velocity 
baseVelstd = std(vBase); % STD of baseline eye velocity
stdsBase = (rhoVeye - baseVel)./baseVelstd; % for each time point, calculate stddev from baseline velocity

pursuit_range  =  stdsBase(motion_onset+50:motion_onset+250);
rxn_time  =  (find((pursuit_range > baseVel+(baseVelstd*2)) == 0, 1, 'last') + 1) + 50;
if ~isempty(rxn_time)
    pursuit_onset  =  rxn_time + motion_onset;
else
    rxn_time = NaN;
    pursuit_onset = NaN;
end

% detect microsaccades
x = (1:length(rhoPeye));
rAcc = (gradient(rhoVeye(:)) ./ gradient(x(:)./1000));
if sum(abs(rAcc(motion_onset-50:motion_onset+50))>750)
    msFlag = 1;
else
    msFlag = 0;
end

% chop traces into 800 ms before - 800 ms after motion onset
time_range = [motion_onset-preint+1 motion_onset+postint];
eye_traces = (cellfun(@(y) (y(time_range(1):time_range(2))'), eye_traces, 'uni', 0))';
target_traces = (cellfun(@(z) (z(time_range(1):time_range(2))'), target_traces, 'uni', 0))';

% f = figure; %('Visible','off');
% plot(x,eye_traces{7},'g-','LineWidth',3)
% hold on
% plot(x,rhoVeye,'k-','LineWidth',2)
% xline(motion_onset,'k--','LineWidth',2)
% xline(pursuit_onset,'k:','LineWidth',2)
% xlabel('time aligned to trial onset (ms)')
% ylabel('radial eye velocity')
% if msFlag==1
%     title('MS')
% else
%     title('no MS')
% end
% xlim([motion_onset-200 pursuit_onset + 500])

end

