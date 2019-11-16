# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from copy import deepcopy


def shift_brightness(X_train):

    s_bright_X = deepcopy(X_train)
    for i in range(X_train.shape[0]):
        random_delta = max(0, np.random.normal(loc=0.15, scale=0.05, size=1))
        s_bright_img = tf.image.adjust_brightness(X_train[i], random_delta)
        s_bright_X[i] = s_bright_img
    return s_bright_X


def random_saturation(X_train):

    r_saturation_X = tf.image.random_saturation(X_train, lower=0.6, upper=1.4)
    return r_saturation_X


def random_distortion(X_train, y_train):

    r_saturation_X = tf.image.random_saturation(X_train, lower=0.6, upper=1.4)
    processed_X_train = np.concatenate((X_train, r_saturation_X), axis=0)
    processed_y_train = np.concatenate((y_train, y_train), axis=0)

    r_contrast_X = tf.image.random_contrast(X_train, lower=1.5, upper=2.0)
    processed_X_train = np.concatenate((processed_X_train, r_contrast_X), axis=0)
    processed_y_train = np.concatenate((processed_y_train, y_train), axis=0)

    r_hue_X = tf.image.random_hue(X_train, max_delta=0.2)
    processed_X_train = np.concatenate((processed_X_train, r_hue_X), axis=0)
    processed_y_train = np.concatenate((processed_y_train, y_train), axis=0)

    return processed_X_train, processed_y_train


if __name__ == '__main__':
    pass
