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
sPath = fullfile('/media/zainkhan/KINGSTON/NiDyN/s16'); % Change this to be your file path
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

%% Save pupil data separately

csvwrite(sprintf('X_pupil_s%i_%s.csv',subject_number,condition), pupil_data);
