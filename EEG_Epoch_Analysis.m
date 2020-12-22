%% NiDyN Data Analysis - EEG P300 Epoch Analysis
close all
clear all

%% Parameters
eeg_srate = 2048;
eye_srate = 120;
unity_srate = 75;
target_category = 4; %camera - REMEMBER THAT THIS IS GOING TO BE DIFFERENT FOR EACH SUBJECT, IT IS EQUIVALENT TO 'TARGET COUNTED' IN THE SUBJECT NOTES FILE
subject_number = 16;
num_eeg_chan = 89;
condition = 'free'; 

%% Load Data
% Load all files of each subject/condition combination
sPath = fullfile('/media/zainkhan/KINGSTON/NiDyN/s16'); % Change this to be your file path
files = dir(sPath);
data = {};
for file = 1:length(files)
    fn = files(file).name;
    if contains(fn,condition)
        data{file} = load_xdf(fullfile(sPath,fn));
    end
end

%% Separate Out EEG Data
eeg_data = {};
eeg_time = {};
for i = 1:length(data)   
    for j = 1:length(data{i})
        if strcmp(data{i}{j}.info.type,'EEG')
            eeg_data{i} = data{i}{j}.time_series;
            eeg_time{i} = data{i}{j}.time_stamps;
        end
    end
end

%% Filter EEG Data from .5-50 Hz - Butterworth Fourth Order
% This is done to take out AC noise (at 60 Hz in the US) and drifts
% (anything below .5 Hz) in the data

% Design the butterworth filter
[bb, aa] = butter(2,[0.5 50]./(eeg_srate/2));
% Check the filter magnitude and phase response
% freqz ( bb, aa, 10000, eeg_srate )

eeg_filt = {};
for i = 1:length(eeg_data)
    if size(eeg_data{i},1) == num_eeg_chan
        for chan = 1:num_eeg_chan
            eeg_filt{i}(chan,:) = filtfilt(bb,aa,double(eeg_data{i}(chan,:)));
        end
    end
end

%% Separate Out Unity Data
unity_data = {};
unity_time = {};
for i = 1:length(data)   
    for j = 1:length(data{i})
        if strcmp(data{i}{j}.info.type,'object_info')
            unity_data{i} = data{i}{j}.time_series;
            unity_time{i} = data{i}{j}.time_stamps;
        end
    end
end

%% Match Unity Time to EEG Time and Add Event Markers to the Last Row of EEG Data
eeg_w_events = {};
for i = 1:length(data)
    if ~isempty(unity_data{i})
        eeg_w_events{i} = cat(1,eeg_filt{i},zeros(1,length(eeg_filt{i})));
        for unity_ind = 1:length(unity_data{i})
            if unity_data{i}(1,unity_ind) ~= 0
                [~,eeg_ind] = min(abs(unity_time{i}(unity_ind) - eeg_time{i}));
                eeg_w_events{i}(end,eeg_ind) = unity_data{i}(1,unity_ind);
            end
        end
    end
end

%% Concatenate All EEG Data (w/ Events) Together
eeg_events_all = [];
for i = 1:length(eeg_w_events)
    eeg_events_all = cat(2,eeg_events_all,eeg_w_events{i});
end

%% Save the EEG with Events Data before EEGLAB
save(sprintf('s%i_eeg_%s.mat',subject_number,condition),'eeg_events_all');

%% EEGLAB
% This is going to be completely new to you. I would go through the rest of
% this section carefully. Go line by line to make sure you understand what
% each line does. 

% Load saved eeg data
load(sprintf('s%i_eeg_%s.mat',subject_number,condition));
% Select only the EEG channels + events for EEGLAB
eeg_events = eeg_events_all(2:65,:); % we recorded only 64 channels of EEG - the first channel is time and the rest (anything above channel 65) are auxillary channels we did not record in this experiment
eeg_events = cat(1,eeg_events,eeg_events_all(end,:)); % we then add event markers as our 65 channel

%% Save the EEG Events + event markers
csvwrite(sprintf('X_s%i_%s.csv',subject_number,condition), eeg_events);

%% Input data into EEGLAB
eeglab redraw; % if unsure if the EEGLAB GUI is up to date - just type this command in the command line
EEG = pop_importdata('dataformat','array','nbchan',0,'data','eeg_events','srate',eeg_srate,'pnts',0,'xmin',0); % import data into EEGLAB
EEG = pop_chanevent(EEG,65,'edge','leading','edgelen',0); % tell EEGLAB which row is the event markers
EEG = pop_editset(EEG, 'chanlocs', 'E:\NiDyN\Preprocessing Files\biosemi64.sph'); % import location file of all the electrodes - change this to be your file path
EEG = eeg_checkset( EEG );

%% Remove the same channels used in ICA analysis - FIX THIS TO BE DONE
% % AUTOMATICALLY
% % Remind me to explain this in person - it would be too complicated to type
% % out the whole explanation for this one
% EEG = pop_select( EEG,'nochannel',{'FC5' 'CP5' 'Oz' 'POz'});
% EEG = eeg_checkset( EEG );
% 
% % Re-reference EEG data
% % Since the biosemi system we used to record our EEG does not have a
% % reference channel we are just using common reference here
% EEG = pop_reref(EEG,[]);
% EEG = eeg_checkset (EEG);
% 
% % Load the saved ICA weights
% % Again I will explain ICA in person but for this purpose it was already
% % done for you
% load(sprintf('s%i_ICA_Weights.mat',subject_number));
% EEG.icaweights = ICA_weights;
% EEG.icasphere = ICA_sphere;
% EEG = eeg_checkset( EEG );
% 
% % Remove ICA components
% % See above
% EEG = pop_subcomp( EEG, [1 2], 0);
% EEG = eeg_checkset( EEG );
% 
% % Epoch the EEG data and remove baseline
% EEG = pop_epoch( EEG, {  }, [-0.5 1.0], 'epochinfo', 'yes'); % epoch data from -0.5-1s from image onset
% EEG = pop_rmbase( EEG, [-200    0]); % baseline data from -0.2-0s from image onset
% EEG = eeg_checkset( EEG );
% 
% % Epoch rejection
% % Here we use different criteria to reject epoch that are outliers from the
% % norm - I would recommend google each command but they basically mark
% % outlier epochs based on threshold, probablity and kurtosis
% [EEG, aThresh] = pop_eegthresh(EEG,1,1:EEG.nbchan,-200,200,EEG.xmin, EEG.xmax - 1/EEG.srate,1,0);
% [EEG, ~, ~, iProb] = pop_jointprob(EEG,1,1:EEG.nbchan,5,5,1,0);
% [EEG, ~, ~, iKurt] = pop_rejkurt(EEG,1,1:EEG.nbchan,5,5,1,0);
% EEG = eeg_rejsuperpose (EEG,1,1,1,1,1,1,1,1);
% EEG = pop_rejepoch(EEG,find(EEG.reject.rejglobal),0);
% EEG = eeg_checkset( EEG );

%% Save Data for Regularized Logistic Regression into Separate Datasets
%{
% Targets - DO THIS THEN START AGAIN FOR DISTRACTORS
event_type = 'targets';
EEG = pop_selectevent( EEG, 'type',1,'renametype','Targets','deleteevents','on','deleteepochs','on','invertepochs','off');
EEG.setname='Targets';
EEG = eeg_checkset( EEG );
fn = sprintf('s%i_eeg_IO_%s_%s.set',subject_number,condition,event_type);
EEG = pop_saveset( EEG, 'filename',fn,'filepath','D:\\NiDyN\\NiDyN_Analysis');
EEG = eeg_checkset( EEG );

% Distractors
event_type = 'distractors';
EEG = pop_selectevent( EEG, 'type',[2:4],'renametype','Distractors','deleteevents','on','deleteepochs','on','invertepochs','off');
EEG.setname='Distractors';
EEG = eeg_checkset( EEG );
fn = sprintf('s%i_eeg_IO_%s_%s.set',subject_number,condition,event_type);
EEG = pop_saveset( EEG, 'filename',fn,'filepath','D:\\NiDyN\\NiDyN_Analysis');
EEG = eeg_checkset( EEG );
%}
%% Separate Epoched Data into Targets and Distractors
eeg_epoch = EEG.data;
event = EEG.event;
for i=1:size(event,2)
    event_type(i) = event(i).type;   
end

%% Save the event types
target_event = event_type == target_category;
csvwrite(sprintf('y_s%i_%s.csv',subject_number,condition), target_event);
%save(sprintf('y_s%i_%s.mat',subject_number,condition), 'target_event');

%% Shape mismatch at the moment
eeg_epoch_filt_targ = eeg_epoch(:,:,event_type == target_category);
eeg_epoch_filt_dist = eeg_epoch(:,:,event_type ~= target_category);

%% Save Epoched Data for Logistic Regression Analysis
%save(sprintf('s%i_eeg_LR_%s.mat',subject_number,condition),'eeg_epoch_filt_targ','eeg_epoch_filt_dist');

%% Plot Average Data for Cz, Pz, POZ
x_axis = linspace(-500,1000,size(eeg_epoch_filt_dist, 2));
% Fz
figure
channel = 34; % Remember that this also changes for each subject since we remove different channels for each subject
Dist = shadedErrorBar(x_axis,mean(eeg_epoch_filt_dist(channel,:,:),3),std(eeg_epoch_filt_dist(channel,:,:),[],3),'-b',1);
hold on
Targ = shadedErrorBar(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3),std(eeg_epoch_filt_targ(channel,:,:),[],3),'-r',1);
plot(x_axis,median(eeg_epoch_filt_dist(channel,:,:),3),'b.')
plot(x_axis,median(eeg_epoch_filt_targ(channel,:,:),3),'r.')
plot(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3)-mean(eeg_epoch_filt_dist(channel,:,:),3),'--k','LineWidth',2)
title('Fz')
xlabel('Time (ms)')
legend([Dist.mainLine, Targ.mainLine], 'distractors', 'targets', 'Location', 'SouthWest')

% Cz
figure
channel = 44; % Remember that this also changes for each subject since we remove different channels for each subject
Dist = shadedErrorBar(x_axis,mean(eeg_epoch_filt_dist(channel,:,:),3),std(eeg_epoch_filt_dist(channel,:,:),[],3),'-b',1);
hold on
Targ = shadedErrorBar(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3),std(eeg_epoch_filt_targ(channel,:,:),[],3),'-r',1);
plot(x_axis,median(eeg_epoch_filt_dist(channel,:,:),3),'b.')
plot(x_axis,median(eeg_epoch_filt_targ(channel,:,:),3),'r.')
plot(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3)-mean(eeg_epoch_filt_dist(channel,:,:),3),'--k','LineWidth',2)
title('Cz')
xlabel('Time (ms)')
legend([Dist.mainLine, Targ.mainLine], 'distractors', 'targets', 'Location', 'SouthWest')

% Pz
figure
channel = 27; % Remember that this also changes for each subject since we remove different channels for each subject
Dist = shadedErrorBar(x_axis,mean(eeg_epoch_filt_dist(channel,:,:),3),std(eeg_epoch_filt_dist(channel,:,:),[],3),'-b',1);
hold on
Targ = shadedErrorBar(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3),std(eeg_epoch_filt_targ(channel,:,:),[],3),'-r',1);
plot(x_axis,median(eeg_epoch_filt_dist(channel,:,:),3),'b.')
plot(x_axis,median(eeg_epoch_filt_targ(channel,:,:),3),'r.')
plot(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3)-mean(eeg_epoch_filt_dist(channel,:,:),3),'--k','LineWidth',2)
title('Pz')
xlabel('Time (ms)')
legend([Dist.mainLine, Targ.mainLine], 'distractors', 'targets', 'Location', 'SouthWest')

% PoZ
figure
channel = 28; % Remember that this also changes for each subject since we remove different channels for each subject
Dist = shadedErrorBar(x_axis,mean(eeg_epoch_filt_dist(channel,:,:),3),std(eeg_epoch_filt_dist(channel,:,:),[],3),'-b',1);
hold on
Targ = shadedErrorBar(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3),std(eeg_epoch_filt_targ(channel,:,:),[],3),'-r',1);
plot(x_axis,median(eeg_epoch_filt_dist(channel,:,:),3),'b.')
plot(x_axis,median(eeg_epoch_filt_targ(channel,:,:),3),'r.')
plot(x_axis,mean(eeg_epoch_filt_targ(channel,:,:),3)-mean(eeg_epoch_filt_dist(channel,:,:),3),'--k','LineWidth',2)
title('CPz')
xlabel('Time (ms)')
legend([Dist.mainLine, Targ.mainLine], 'distractors', 'targets', 'Location', 'SouthWest')



        


