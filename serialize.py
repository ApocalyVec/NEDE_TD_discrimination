from dask import dataframe as dd
import os
import numpy as np
import pickle

conditions = ['eye', 'free']
subjects = ['s12']
data_root = '/home/apocalyvec/Data/NEDE_target_distractor/'

for cndt in conditions:
    for sbj in subjects:
        x_path = os.path.join(data_root, 'X_' + sbj + '_' + cndt + '.csv')
        y_path = os.path.join(data_root, 'y_' + sbj + '_' + cndt + '.csv')
        # reading input data
        print('Reading ' + y_path + '...')
        Y = dd.read_csv(y_path, header=None).to_records().compute()
        print('Reading ' + x_path + ', this might take a while...')
        data = dd.read_csv(x_path, header=None).to_records().compute()

        print('Dumping pickle')
        np.save(x_path.replace('.csv', '.p'), data)
        np.save(y_path.replace('.csv', '.p'), Y)