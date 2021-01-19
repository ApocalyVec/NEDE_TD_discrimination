close all;
clear all;

%% Set parameters

eeg_srate = 2048;
eye_srate = 120;
unity_srate = 75;
target_category = 4; %camera - REMEMBER THAT THIS IS GOING TO BE DIFFERENT FOR EACH SUBJECT, IT IS EQUIVALENT TO 'TARGET COUNTED' IN THE SUBJECT NOTES FILE
subject_number = 16;
num_eeg_chan = 89;
condition = 'free'; 

%% Load data

% Load all files of each subject/condition combination
sPath = fullfile(strcat('/media/zainkhan/KINGSTON/NiDyN/s', int2str(subject_number))); % Change this to be your file path
files = dir(sPath);
data = {};
for file = 1:length(files)
    fn = files(file).name;
    if contains(fn,condition)
        data{file} = load_xdf(fullfile(sPath,fn));
    end
end


%% Separate Out Pupil Data

pupil_data = {};
pupil_time = {};
for i = 1:length(data)   
    for j = 1:length(data{i})
        if strcmp(data{i}{j}.info.type,'PupilData')
            pupil_data{i} = data{i}{j}.time_series;
            pupil_time{i} = data{i}{j}.time_stamps;
        end
    end
end

%% Downsample Pupil Data

scale = 6; % Factor of 6 change from 120 Hz to 20 Hz
pupil_data_down = {};
pupil_time_down = {};
for i = 1:length(pupil_data)
    pupil_data_down{i} = downsample((pupil_data{i})', scale);
    pupil_data_down{i} = pupil_data_down{i}';
    pupil_time_down{i} = downsample((pupil_time{i})', scale);
    pupil_time_down{i} = pupil_time_down{i}';
end

%% Extract Unity Data

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

%% Align Unity and Pupil Data

pupil_w_events = {};
for i = 1:length(data)
    if ~isempty(unity_data{i})
        pupil_w_events{i} = cat(1,pupil_data_down{i},zeros(1,length(pupil_data_down{i})));
        for unity_ind = 1:length(unity_data{i})
            if unity_data{i}(1,unity_ind) ~= 0
                [~,pupil_ind] = min(abs(unity_time{i}(unity_ind) - pupil_time_down{i}));
                pupil_w_events{i}(end,pupil_ind) = unity_data{i}(1,unity_ind);
            end
        end
    end
end


%% Concatanete all Pupil Data (w/ Events)

pupil_events_all = [];
for i = 1:length(pupil_w_events)
    pupil_events_all = cat(2,pupil_events_all,pupil_w_events{i});
end

%% Interpolate Pupil Data

pupil_interp = fillmissing(pupil_events_all, 'linear');

%% Save pupil data separately

csvwrite(sprintf('pupil_s%i_%s.csv',subject_number,condition), pupil_interp);
