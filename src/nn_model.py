# -*- coding: utf-8 -*-
import tensorflow as tf


def lenet():

    model = tf.keras.models.Sequential()

    # Layer 1: Convolutional. Input:32x32x3, Output:26x26x6. (filters=6, kernel_size=(5, 5))
    model.add(tf.keras.layers.Conv2D(6, (5, 5), padding='VALID', activation='relu', input_shape=(32, 32, 3)))

    # Layer 2: Pooling. Input:26x26x6, Output:14x14x6. (pool_size=(2, 2), strides=(2, 2))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='VALID'))

    # Layer 3: Convolutional. Input:14x14x6, Output:10x10x16. (filters=16, kernel_size=(5, 5))
    model.add(tf.keras.layers.Conv2D(16, (5, 5), padding='VALID', activation='relu'))

    # Layer 4: Pooling. Input:10x10x16, Output:5x5x16. (pool_size=(2, 2), strides=(2, 2))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='VALID'))

    # Layer 5: Fully Connected. Input:5x5x16, Output:400.
    model.add(tf.keras.layers.Flatten())

    # Layer 5: Fully Connected. Input:400, Output:120.
    model.add(tf.keras.layers.Dense(120, activation='relu'))

    # Layer 6: Fully Connected. Input:120, Output:84.
    model.add(tf.keras.layers.Dense(84, activation='relu'))

    # Layer 5: Fully Connected. Input:84, Output:43.
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    return model


def proposed():


    model = tf.keras.models.Sequential()

    # Layer 1: Convolutional. Input:32x32x3, Output:30x30x6. (filters=6, kernel_size=(3, 3))
    model.add(tf.keras.layers.Conv2D(6, (3, 3), padding='VALID', activation='relu', input_shape=(32, 32, 3)))

    # Layer 2: Pooling. Input:30x30x6, Output:28x28x6. (pool_size=(3, 3), strides=(1, 1))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), (1, 1), padding='VALID'))

    # Layer 3: Convolutional. Input:28x28x6, Output:24x24x11. (filters=14, kernel_size=(5, 5))
    model.add(tf.keras.layers.Conv2D(14, (5, 5), padding='VALID', activation='relu'))

    # Layer 4: Pooling. Input:24x24x11, Output:12x12x11. (pool_size=(2, 2), strides=(2, 2))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='VALID'))

    # Layer 5: Convolutional. Input:12x12x11. Output:6x6x18. (filters=22, kernel_size=(7, 7))
    model.add(tf.keras.layers.Conv2D(22, (7, 7), padding='VALID', activation='relu'))

    # Layer 6: Pooling. Input:6x6x18, Output:5x5x22. (pool_size(2, 2), strides=(1, 1))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (1, 1), padding='VALID'))

    # Layer 7: Fully Connected. Input:5x5x22, Output:550.
    model.add(tf.keras.layers.Flatten())

    # Layer 8: Fully Connected. Input:550, Output:150.
    model.add(tf.keras.layers.Dense(150, activation='relu'))

    # Layer 9: Fully Connected. Input:150, Output:100.
    model.add(tf.keras.layers.Dense(100, activation='relu'))

    # Layer 10: Fully Connected. Input:100, Output:43.
    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    return model


if __name__ == '__main__':
    print("[test]: nn_model")
    model = lenet()
    model.summary()

    model = proposed()
    model.summary()
