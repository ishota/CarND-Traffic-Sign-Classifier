# -*- coding: utf-8 -*-
import tensorflow
import os
from pathlib import Path
from load_pickled_data import *
from summary_data import *
from nn_model import *


def main():

    # load_pickled_data
    data_path = str(Path('.').resolve().parents[0]) + os.sep + 'traffic-signs-data'
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pickled_data(data_path)

    # summary_data
    csv_file = str(Path('.').resolve().parents[0]) + os.sep + 'signnames.csv'
    n_classes = summary_data(X_train, y_train, X_valid, y_valid, X_test, y_test, csv_file)

    # Training Pipeline
    logits = LeNet()
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.one_hot(n_classes))
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    training_operation = optimizer.minimize(loss_operation)


if __name__ == '__main__':
    main()
