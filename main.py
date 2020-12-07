from dask import dataframe as dd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from utils import build_train_birnn_with_attention, plot_train_history, plot_roc_multiclass, window_slice


def encode_y(Y):
    # one-hot encode integer-valued y in the shape (1, num_samples) or (num_samples, 1)
    assert len(Y.shape) == 2
    assert Y.shape[0] == 1 or Y.shape[1] == 1
    Y = np.transpose(Y) if Y.shape[0] == 1 else Y

    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(Y).toarray()
    return Y_encoded, encoder

conditions = ['free', 'eye']
subjects = ['s12']
data_root = '/home/apocalyvec/Data/NEDE_target_distractor/'

np.random.seed(42)
f_sample = 2048
start = 0.2  # in seconds
end = 0.6
timesteps_per_sample = int(f_sample * (end - start))
num_channels = 64

scenario_train_histories = dict()
data_all = np.empty((0, num_channels))
X_all = np.empty((0, timesteps_per_sample, num_channels))
Y_all = np.empty((0, 1))
x_event_time_indices_all = np.empty((0,))

for cndt in conditions:
    for sbj in subjects:
        x_path = os.path.join(data_root, 'X_' + sbj + '_' + cndt + '.npy')
        y_path = os.path.join(data_root, 'y_' + sbj + '_' + cndt + '.npy')
        # reading input data
        print('Reading ' + y_path)
        Y = np.load(y_path)
        Y = np.transpose(Y)
        Y_all = np.concatenate([Y_all, Y])
        Y_encoded = encode_y(Y)

        print('Reading ' + x_path)
        data = np.load(x_path)
        data_all = np.concatenate([data_all, data])

        print('Processing samples')
        x_event_vector = data[-1]
        x_event_vector = np.array(x_event_vector, dtype=int)
        x_event_time_indices = np.where(x_event_vector != 0)[0]  # same dim as Y
        data = np.transpose(data[:num_channels])
        X = np.array([data[list(range(i + int(start * f_sample), i + int(end * f_sample))), :] for i in x_event_time_indices])

        x_event_time_indices_all = np.concatenate([x_event_time_indices_all, x_event_time_indices])
        X_all = np.concatenate([X_all, X])

        x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.20, random_state=3, shuffle=True)

        # test the BIRNN_attention
        model_name = 'BIRNN attention with attention'
        history = build_train_birnn_with_attention(x_train, x_test, y_train, y_test, note=sbj + '-' + cndt + '\n' + model_name)
        eval = history.model.evaluate(x=x_test, y=y_test)
        scenario_train_histories[('BIRNN_attention', sbj, cndt)] = [history, eval]

print('Evaluating all data')
Y_all, _ = encode_y(Y_all)
x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.20, random_state=3, shuffle=True)
model_name = 'BIRNN attention with attention'
history = build_train_birnn_with_attention(x_train, x_test, y_train, y_test, note='s12' + model_name)

# evaluate events
s_start = -0.5
s_end = 1.2
event_time_index = int(x_event_time_indices_all[np.random.choice(np.where(Y_all == 1)[0])])  # find the time index of a target
data_slice = data_all[list(range(event_time_index + int(s_start * f_sample), event_time_index + int(s_end * f_sample))), :]
samples = window_slice(data_slice, timesteps_per_sample, 1)
sample_t_vector = np.linspace(s_start + start, s_end, len(data_slice) - timesteps_per_sample)
sample_pred = history.model.predict(samples, batch_size=32)
a = np.argmax(sample_pred, axis=1)
# plt.plot(sample_t_vector, sample_pred[:,0], label='P(TARGET)')
# plt.plot(sample_t_vector, sample_pred[:,1], label='P(DISTRACTOR)')
plt.plot(sample_t_vector, a, label='LABEL=TARGET')
plt.ylabel('Predicted Labels')
# plt.ylabel('Predicted probability of TARGET & DISTRACTOR')
plt.xlabel('Time before and after TARGET event (sec)')
plt.legend(loc='center right')
plt.show()