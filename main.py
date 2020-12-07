import numpy as np
import pandas


duration = 100
f_sample = 2048
num_channel = 64
data = np.random.random((duration * f_sample, num_channel))
label = np.zeros((duration * f_sample, num_channel))

# load in data

