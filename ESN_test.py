import tensorflow as tf
from tensorflow_addons.layers import ESN
from tensorflow_addons.rnn import ESNCell
from typeguard import typechecked
import numpy as np

data = np.random.random((200, 819, 64))

sequence_input = tf.keras.layers.Input(shape=(819, 64))
x = ESN(units=256)(sequence_input)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.05)(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

classifier = tf.keras.Model(inputs=sequence_input, outputs=outputs)
adam = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-7)
classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
# fit to data
history = classifier.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                         epochs=epochs, batch_size=batch_size, callbacks=[es])