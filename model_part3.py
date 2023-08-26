import tensorflow as tf
import numpy as np
import copy

def linear_unit(x, W, b):
  return tf.matmul(x, W) + b


INPUT_CHANNELS = 3

class ModelPart3:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """
        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        """
        CNN-1 filter parameters
        """
        self.filter_height = 5
        self.filter_width = 5
        self.output_channels = 16
        self.strides = [1, 1, 1, 1]

        self.filters = tf.Variable(
            tf.random.truncated_normal([self.filter_height, self.filter_width,
                                        INPUT_CHANNELS, self.output_channels],
                                        dtype = tf.float32, stddev = 0.1),
                                        name = "filters")
        self.cnn_bias = tf.Variable(tf.random.normal([self.output_channels]))

        """
        Max-pooling
        """
        self.pool_size = 2
        self.pool_strides = [1, 2, 2, 1]

        cnn_input = 32 * 32 * self.output_channels
        layer_1_output = 256
        output = 2

        self.W1 = tf.Variable(
            tf.random.truncated_normal([cnn_input, layer_1_output],
                                       dtype=tf.float32, stddev=0.1),
            name="W1")
        self.B1 = tf.Variable(
            tf.random.truncated_normal([1, layer_1_output],
                                       dtype=tf.float32, stddev=0.1),
            name="B1")

        self.W2 = tf.Variable(
            tf.random.truncated_normal([layer_1_output, output],
                                       dtype=tf.float32, stddev=0.1),
            name="W1")
        self.B2 = tf.Variable(
            tf.random.truncated_normal([1, output],
                                       dtype=tf.float32, stddev=0.1),
            name="B1")

        self.trainable_variables = [self.filters, self.cnn_bias,
                                    self.W1, self.B1, self.W2, self.B2]
        self.optimal_variables = [self.filters, self.cnn_bias,
                                  self.W1, self.B1, self.W2, self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # shape of input  (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of output (num_inputs, output_height, output_width, output_channels)
        x0_conv = tf.nn.conv2d(inputs, self.filters, [1, 1, 1, 1], padding="SAME")
        x0_conv_bias = tf.nn.bias_add(x0_conv, self.cnn_bias)
        x0_relu = tf.nn.relu(x0_conv_bias)

        # pool1
        # pool1 = tf.nn.max_pool(x0_relu, ksize=[1, self.pool_size, self.pool_size, 1],
        #                        strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # this reshape "flattens" the image data
        x0_inputs = tf.reshape(x0_relu, [x0_relu.shape[0],-1])

        x1 = linear_unit(x0_inputs, self.W1, self.B1)
        # apply ReLU activation
        x1_relu = tf.nn.relu(x1)
        x2 = linear_unit(x1_relu, self.W2, self.B2)

        return x2
    
    def update_optimal_variables(self):
        self.optimal_variables = copy.deepcopy(self.trainable_variables)
