# -*- coding: utf-8 -*-
import tensorflow as tf


def LeNet(X):
    # Hyperparameters
    # for normal
    mu = 0
    sigma = 1

    #Convolutional. Input = 32x32x1, Output = 28x28x6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(X, conv1_W, strides=(1, 1, 1, 1), padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6, Output = 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x14x6, Output = 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=(1, 1, 1, 1), padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16, Output = 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')

    # Flatten. Input = 5x5x16, Output = 400
    fc3 = tf.reshape(conv2, -1)

    # Fully Connected. Input = 400, Output = 120
    fc4_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(120))
    fc4 = tf.add(tf.matmul(fc3, fc4_W), fc4_b)
    fc4 = tf.nn.relu(fc4)

    # Fully Connected. Input = 120, Output = 84
    fc5_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc5_b = tf.Variable(tf.zeros(84))
    fc5 = tf.add(tf.matmul(fc4, fc5_W), fc5_b)
    fc5 = tf.nn.relu(fc5)

    # Fully Connected. Input = 84, Output = 43
    fc6_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc6_b = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc5, fc6_W), fc6_b)

    return logits


if __name__ == '__main__':
    print("[test]: nn_model")
