import pandas as pd
import os
import numpy as np
import pickle
import time
from multiprocessing import Pool, freeze_support
import multiprocessing


def print_names(cndt, sbj):
    print('Processing ' + cndt + ' ' + sbj)


def re_serialize_npy(cndt, sbj, data_root):
    print('Processing ' + cndt + ' ' + sbj)
    start = time.time()
    x_path = os.path.join(data_root, 'X_' + sbj + '_' + cndt + '.csv')
    y_path = os.path.join(data_root, 'y_' + sbj + '_' + cndt + '.csv')
    # reading input data
    print('Reading ' + y_path + '...')
    Y = pd.read_csv(y_path, header=None).values
    print('Dumping pickle')
    np.save(y_path.replace('.csv', ''), Y)

    print('Reading ' + x_path + ', this might take a while...')
    data = pd.read_csv(x_path, header=None).values
    print('Dumping pickle')
    np.save(x_path.replace('.csv', ''), data)

    print('Took' + str(time.time() - start))

if __name__ == '__main__':
    conditions = ['eye', 'free']
    subjects = ['s' + str(i) for i in range(8, 16 - 1)]
    data_root = '/Users/Leo/Dropbox/data/NEDE_TD_discrimination/Data/'

    freeze_support()

    arg_list = [(cndt, sbj, data_root) for cndt in conditions for sbj in subjects]
    pool = multiprocessing.Pool(4)
    pool.starmap(re_serialize_npy, arg_list)
    print('All done')
# for cndt in conditions:
#     for sbj in subjects:
#         re_serialize_npy(cndt, sbj)

