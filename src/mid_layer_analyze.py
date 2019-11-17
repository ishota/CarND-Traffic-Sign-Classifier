# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


def layer_analyze(nn_model, input_images, images):

    # Layer : conv2d
    mid_layer1_model = tf.keras.models.Model(inputs=nn_model.input, outputs=nn_model.get_layer('conv2d').output)
    mid_layer1_output = mid_layer1_model.predict(input_images)

    # Layer : conv2d_1
    mid_layer2_model = tf.keras.models.Model(inputs=nn_model.input, outputs=nn_model.get_layer('conv2d_1').output)
    mid_layer2_output = mid_layer2_model.predict(input_images)

    # Layer : conv2d_2
    mid_layer3_model = tf.keras.models.Model(inputs=nn_model.input, outputs=nn_model.get_layer('conv2d_2').output)
    mid_layer3_output = mid_layer3_model.predict(input_images)

    img_idx = 0

    # plot layer
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0)
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(images[0])
    ax.set_axis_off()
    ax = fig.add_subplot(1, 4, 2)
    ax.matshow(mid_layer1_output[img_idx, :, :, 0])
    ax.set_axis_off()
    ax = fig.add_subplot(1, 4, 3)
    ax.matshow(mid_layer2_output[img_idx, :, :, 0])
    ax.set_axis_off()
    ax = fig.add_subplot(1, 4, 4)
    ax.matshow(mid_layer3_output[img_idx, :, :, 0])
    ax.set_axis_off()
    plt.show()

    # layer 1
    layer_output(mid_layer1_output, img_idx)

    # layer 2
    layer_output(mid_layer2_output, img_idx)

    # layer 3
    layer_output(mid_layer3_output, img_idx)


def layer_output(layer, img_idx):
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0)
    for i in range(layer.shape[3]):
        ax = fig.add_subplot(1, layer.shape[3], i+1)
        ax.matshow(layer[img_idx, :, :, i])
        ax.set_axis_off()
    plt.show()


if __name__ == '__main__':
    pass
