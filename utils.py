import datetime
import os

import numpy as np
import matplotlib.pylab as plt
from tensorflow.python.keras.backend import clear_session
import tensorflow as tf
from matplotlib.pyplot import cm
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import CuDNNLSTM, Concatenate

from attention import Attention


def window_slice(data, window_size, stride):
    assert window_size <= len(data)
    assert stride > 0
    rtn = np.expand_dims(data, axis=0) if window_size == len(data) else []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)

def build_train_rnn(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()
    classifier = tf.keras.Sequential()
    classifier.add(CuDNNLSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    classifier.add(CuDNNLSTM(units=64, return_sequences=True, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))  # ignore 20% of the neurons in both forward and backward propagation
    classifier.add(CuDNNLSTM(units=64, return_sequences=False, kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)))
    classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(units=128, kernel_initializer='random_uniform'))
    classifier.add(tf.keras.layers.Dropout(0.2))
    classifier.add(tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform'))
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

def build_train_ann(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()

    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1:])))
    classifier.add(tf.keras.layers.Dense(units=128, kernel_initializer='random_uniform'))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))

    classifier.add((tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='random_uniform')))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))
    classifier.add((tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform')))

    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history


def build_train_cnn(x_train, x_test, y_train, y_test, epochs=250, batch_size=64):
    clear_session()
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), input_shape=(x_train.shape[1:]), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Conv1D(filters=16, kernel_size=(3, ), kernel_initializer='random_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=1e-5)))
    classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    classifier.add(tf.keras.layers.Flatten())

    classifier.add((tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='random_uniform')))
    classifier.add(tf.keras.layers.Dropout(rate=0.2))
    classifier.add((tf.keras.layers.Dense(units=y_train.shape[1], activation='softmax', kernel_initializer='random_uniform')))

    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    return history


def build_train_birnn_with_attention(x_train, x_test, y_train, y_test, note='', epochs=300, patience=50, batch_size=32):
    clear_session()

    sequence_input = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(256, return_sequences=True), name="bi_lstm_0")(sequence_input)
    lstm = tf.keras.layers.Dropout(0.2, name="drop_0")(lstm)
    # lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(512, return_sequences=True), name="bi_lstm_1")(lstm)
    # lstm = tf.keras.layers.Dropout(0.2, name="drop_1")(lstm)
    # lstm = tf.keras.layers.Bidirectional(CuDNNLSTM(512, return_sequences=True), name="bi_lstm_2")(lstm)
    # lstm = tf.keras.layers.Dropout(0.2, name="drop_2")(lstm)
    (lstm, forward_h, forward_c, backward_h, backward_c) = tf.keras.layers.Bidirectional(
        CuDNNLSTM(256, return_sequences=True, return_state=True), name="bi_lstm_1")(lstm)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    context_vector, attention_weights = Attention(10)(lstm, state_h)
    dense1 = tf.keras.layers.Dense(256, activation="relu")(context_vector)
    dropout = tf.keras.layers.Dropout(0.05)(dense1)
    output = tf.keras.layers.Dense(y_train.shape[1], activation="softmax")(dropout)

    classifier = tf.keras.Model(inputs=sequence_input, outputs=output)
    adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
    classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                             epochs=epochs, batch_size=batch_size, callbacks=[es])
    # plot training history
    plot_train_history(history, note=note)

    # plot ROC
    y_score = history.model.predict(x_test)
    fig, ax = plt.subplots()
    plot_roc_multiclass(n_classes=y_train.shape[1], y_score=y_score, y_test=y_test, ax=ax, zoom=False)
    ax.set_title('ROC for ' + note)
    ax.legend(loc="lower right")
    plt.show()

    return history

def plot_train_history(history, note=''):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy ' + note)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss ' + note)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_cm_results(freqs, locs, train_histories, note=''):
    acc_matrix = np.array([entry[1][1] for scn, entry in train_histories]).reshape((3, 3))

    fig, ax = plt.subplots()
    im = ax.imshow(acc_matrix,vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(freqs)))
    ax.set_yticks(np.arange(len(locs)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(freqs, size=18)
    ax.set_yticklabels(locs, size=18)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(locs)):
        for j in range(len(freqs)):
            text = ax.text(j, i, round(acc_matrix[i, j], 3),
                           ha="center", va="center", color="black", size=20)

    ax.set_title("Three motion classification accuracies " + note, size=15)
    # fig.tight_layout()
    plt.show()

import io
import cv2

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

def plot_roc_multiclass(n_classes, y_score, y_test, ax=None, zoom=False):
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()

    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if ax is None:
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

        # Zoom in view of the upper left corner.
        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
    else:
        if zoom:
            # Zoom in view of the upper left corner.
            ax.set_xlim(0, 0.2)
            ax.set_ylim(0.8, 1)
            ax.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            ax.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")

        else:
            ax.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            ax.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            ax.plot([0, 1], [0, 1], 'k--', lw=lw)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")

