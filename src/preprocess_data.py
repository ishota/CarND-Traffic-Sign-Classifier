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


if __name__ == '__main__':
    pass
