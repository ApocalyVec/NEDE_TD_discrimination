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
    data_path = os.path.join(data_root, 'P_data_with_event_' + sbj + '_' + cndt + '.csv')
    data_time_path = os.path.join(data_root, 'P_time_' + sbj + '_' + cndt + '.csv')
    # reading input data
    print('Reading ' + data_time_path + '...')
    data_time = pd.read_csv(data_time_path, header=None).values
    print('Dumping pickle')
    np.save(data_time_path.replace('.csv', ''), data_time)

    print('Reading ' + data_path + ', this might take a while...')
    data = pd.read_csv(data_path, header=None).values
    print('Dumping pickle')
    np.save(data_path.replace('.csv', ''), data)

    print('Took' + str(time.time() - start))

if __name__ == '__main__':
    conditions = ['eye', 'free']
    subjects = ['s' + str(i) for i in range(8, 17)]
    data_root = '/home/apocalyvec/Dropbox/data/NEDE_TD_discrimination/Pupil_Data'

    freeze_support()

    arg_list = [(cndt, sbj, data_root) for cndt in conditions for sbj in subjects]
    pool = multiprocessing.Pool(16)
    pool.starmap(re_serialize_npy, arg_list)
    print('All done')
# for cndt in conditions:
#     for sbj in subjects:
#         re_serialize_npy(cndt, sbj)

