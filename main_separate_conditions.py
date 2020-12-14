import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from data_utils import load_data
from utils import build_train_birnn_with_attention, plot_train_history, plot_roc_multiclass, window_slice, \
    build_transformer_multiheaded, build_ESN



conditions = [['free'], ['eye']]
subjects = ['s' + str(i) for i in range(14, 16 - 1)]
data_root = '/media/apocalyvec/Samsung PSS/Data/'
model_name_train_callback_dict = {'Multiheaded Transformer': build_transformer_multiheaded, 'Echo State Network': build_ESN, 'BiLSTM with attention': build_train_birnn_with_attention}
scenario_train_histories = dict()

for cnd in conditions:
    x_train, x_test, y_train, y_test, encoder = load_data(data_root, cnd, subjects)
    for model_name, train_callback in model_name_train_callback_dict.items():
        # fit to models
        scenario = 'All subjects, ' + str(cnd[0])
        history, clsf_rpt = train_callback(x_train, x_test, y_train, y_test, encoder, patience=25, note=model_name + ' ' + scenario)
        scenario_train_histories[model_name + ' ' + scenario] = [history, clsf_rpt]
