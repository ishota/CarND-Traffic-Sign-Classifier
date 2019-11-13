# -*- coding: utf-8 -*-
import tensorflow as tf


def lenet():

    # input shape
    X = (32, 32, 3)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='VALID', activation='relu', input_shape=X))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='VALID', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    return model


if __name__ == '__main__':
    print("[test]: nn_model")
    model = lenet()
    model.summary()
