import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from utils import build_train_birnn_with_attention, plot_train_history, plot_roc_multiclass, window_slice, \
    build_transformer_multiheaded, build_ESN, build_ESN_two_input, build_transformer_multiheaded_two_input, \
    build_train_birnn_with_attention_two_input


def encode_y(Y):
    # one-hot encode integer-valued y in the shape (1, num_samples) or (num_samples, 1)
    assert len(Y.shape) == 2
    assert Y.shape[0] == 1 or Y.shape[1] == 1
    Y = np.transpose(Y) if Y.shape[0] == 1 else Y

    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(Y).toarray()
    return Y_encoded, encoder


conditions = ['free', 'eye']
subjects = ['s' + str(i) for i in range(8, 16-1)]

data_root_eeg = '/home/apocalyvec/Dropbox/data/NEDE_TD_discrimination/Data/'
data_root_eye = '/home/apocalyvec/Dropbox/data/NEDE_TD_discrimination/Pupil_Data/'

np.random.seed(42)  # set random seed

# EEG parameters
eeg_f_sample = 2048
eeg_start = 0.2  # in seconds
eeg_end = 0.6
eeg_num_channels = 64
eeg_timesteps_per_sample = int(eeg_f_sample * (eeg_end - eeg_start))

# Eye parameters
eye_f_sample = 120
eye_start = -1.  # in seconds
eye_end = 3.
eye_timesteps_per_sample = int(eye_f_sample * (eye_end - eye_start))
eye_num_channels = 10
eye_pupil_channels = [0, 5]

# Data arrays
scenario_train_histories = dict()
data_all_eeg = np.empty((eeg_num_channels + 1, 0))  # plus 1 for event markers
data_all_eye = np.empty((eye_num_channels, 0))

X_all_eeg = np.empty((0, eeg_timesteps_per_sample, eeg_num_channels))
X_all_eye = np.empty((0, eye_timesteps_per_sample, 2))

Y_all = np.empty((0, 1))

x_event_time_indices_all_eeg = np.empty((0,))
x_event_time_indices_all_eye = np.empty((0,))

for cndt in conditions:
    for sbj in subjects:
        # load in the eeg data #################################################
        x_path = os.path.join(data_root_eeg, 'X_' + sbj + '_' + cndt + '.npy')
        y_path = os.path.join(data_root_eeg, 'y_' + sbj + '_' + cndt + '.npy')
        # reading input data
        print('Reading ' + y_path)
        Y = np.load(y_path)
        Y = np.transpose(Y)[:-1]  # discard the last eye
        Y_all = np.concatenate([Y_all, Y])
        Y_encoded = encode_y(Y)

        print('Reading ' + x_path)
        data_eeg = np.load(x_path)
        data_all_eeg = np.concatenate([data_all_eeg, data_eeg], axis=1)

        print('Processing samples')
        x_event_vector = data_eeg[-1]
        x_event_vector = np.array(x_event_vector, dtype=int)
        x_event_time_indices_eeg = np.where(x_event_vector != 0)[0][:-1]  # discard the last eye  # same dim as Y
        data_eeg = np.transpose(data_eeg[:eeg_num_channels])
        X_eeg = np.array(
            [data_eeg[list(range(i + int(eeg_start * eeg_f_sample), i + int(eeg_end * eeg_f_sample))), :] for i in
             x_event_time_indices_eeg])

        x_event_time_indices_all_eeg = np.concatenate([x_event_time_indices_all_eeg, x_event_time_indices_eeg])
        X_all_eeg = np.concatenate([X_all_eeg, X_eeg])

        # load in the eye data #################################################
        x_path = os.path.join(data_root_eye, 'P_data_with_event_' + sbj + '_' + cndt + '.npy')
        data_eye = np.load(x_path)
        x_event_time_indices_eye = np.where(data_eye[-1, :] != 0)[0][:-1]  # discard the last eye
        data_eye = data_eye[:eye_num_channels]
        data_all_eye = np.concatenate([data_all_eye, data_eye], axis=1)
        X_eye = np.array(
            [np.transpose(data_eye)[list(range(i + int(eye_start * eye_f_sample), i + int(eye_end * eye_f_sample))), :]
             for i in x_event_time_indices_eye])
        x_event_time_indices_all_eye = np.concatenate([x_event_time_indices_all_eye, x_event_time_indices_eye])
        X_eye = X_eye[:, :, eye_pupil_channels]
        X_all_eye = np.concatenate([X_all_eye, X_eye])
        pass
        # x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.20, random_state=3, shuffle=True)

        # test the BIRNN_attention
        # model_name = 'BIRNN attention with attention'
        # history = build_train_birnn_with_attention(x_train, x_test, y_train, y_test, note=sbj + '-' + cndt + '\n' + model_name)
        # eval = history.model.evaluate(x=x_test, y=y_test)
        # scenario_train_histories[('BIRNN_attention', sbj, cndt)] = [history, eval]

print('Evaluating all data')

# min-max normalize x
X_all_eeg = (X_all_eeg - np.min(X_all_eeg)) / (np.max(X_all_eeg) - np.min(X_all_eeg))
X_all_eye = (X_all_eye - np.min(X_all_eye)) / (np.max(X_all_eye) - np.min(X_all_eye))

Y_all, encoder = encode_y(Y_all)
x_train_eeg, x_test_eeg, y_train, y_test = train_test_split(X_all_eeg, Y_all, test_size=0.20, random_state=3,
                                                            shuffle=True)
x_train_eye, x_test_eye, y_train, y_test = train_test_split(X_all_eeg, Y_all, test_size=0.20, random_state=3,
                                                            shuffle=True)

# fit to models
model_name_train_callback_dict = {
    'BiLSTM with attention': build_train_birnn_with_attention_two_input,
    'Multiheaded Transformer': build_transformer_multiheaded_two_input,
    'Echo State Network': build_ESN_two_input,
}
scenario = 'All subjects, All conditions'
for model_name, train_callback in model_name_train_callback_dict.items():
    history, clsf_rpt = train_callback(x_train_eeg, x_train_eye, x_test_eeg, x_test_eye, y_train, y_test, encoder,
                                       note=model_name + ' ' + scenario,
                                       patience=25)
    scenario_train_histories[model_name + ' ' + scenario] = [history, clsf_rpt]

    # # evaluate events
    # s_start = -0.5
    # s_end = 1.2
    # event_time_index = int(
    #     x_event_time_indices_all[np.random.choice(np.where(Y_all == 1)[0])])  # find the time index of a target
    # data_slice = data_all[:num_channels].transpose()[
    #              list(range(event_time_index + int(s_start * f_sample), event_time_index + int(s_end * f_sample))), :]
    # samples = window_slice(data_slice, timesteps_per_sample, 1)
    # sample_t_vector = np.linspace(s_start + start, s_end, len(data_slice) - timesteps_per_sample)
    # sample_pred = history.model.predict(samples, batch_size=32)
    # a = np.argmax(sample_pred, axis=1)
    # plt.plot(sample_t_vector, sample_pred[:, 0], label='P(DISTRACTOR)')
    # plt.plot(sample_t_vector, sample_pred[:, 1], label='P(TARGET)')
    # # plt.plot(sample_t_vector, a, label='LABEL=TARGET')
    # # plt.ylabel('Predicted Labels')
    # plt.ylabel('Predicted probability of TARGET & DISTRACTOR')
    # plt.xlabel('Time before and after TARGET event (sec)')
    # plt.legend(loc='lower right')
    # plt.show()
