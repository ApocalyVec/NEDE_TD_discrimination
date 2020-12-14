import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


if __name__ == '__main__':
    data_root = '/media/apocalyvec/Samsung PSS/Data/'
    conditions = ['free', 'eye']
    subjects = ['s' + str(i) for i in range(8, 16 - 1)]