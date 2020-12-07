from dask import dataframe as dd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

conditions = ['eye', 'free']
subjects = ['s12']
data_root = '/home/apocalyvec/Data/NEDE_target_distractor/'

# f_sample = 2048
start = 0.2  # in seconds
end = 0.6
num_channel = 64

for cndt in conditions:
    for sbj in subjects:
        x_path = os.path.join(data_root, 'X_' + sbj + '_' + cndt + '.p')
        y_path = os.path.join(data_root, 'y_' + sbj + '_' + cndt + '.p')
        # reading input data
        print('Reading ' + y_path + '...')
        Y = pickle.load(open(y_path, 'rb'))
        print('Reading ' + x_path + ', this might take a while...')
        data = pickle.load(open(x_path, 'rb'))

        print('finished reading')
        x_event_vector = data[-1]
        print(x_event_vector[:300])

        x_event_vector = np.array(x_event_vector, dtype=int)
        x_event_indices = np.where(x_event_vector != 0)[0]
        X = np.array([data[:64, list(range(i + start, i + end))] for i in x_event_indices])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=3, shuffle=True)

        break

# load in data

