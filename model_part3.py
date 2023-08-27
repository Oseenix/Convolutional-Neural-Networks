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
        self.batch_size = 128
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        """
        CNN-1 filter parameters
        """
        self.filter_height = 5
        self.filter_width = 5
        self.output_channels = 8
        self.strides = [1, 1, 1, 1]

        """
        Max-pooling
        """
        self.pool_size = 3
        self.pool_strides = [2, 2]


        self.filters = tf.Variable(
            tf.random.truncated_normal([self.filter_height, self.filter_width,
                                        INPUT_CHANNELS, self.output_channels],
                                        dtype = tf.float32, stddev = 0.1),
                                        name = "filters")
        self.cnn_bias = tf.Variable(tf.random.normal([self.output_channels]))


        """
        CNN-2 filter parameters
        self.filter_height_2 = 3
        self.filter_width_2 = 3
        self.output_channels_2 = 32
        self.strides_2 = [1, 1, 1, 1]

        self.filters_2 = tf.Variable(
            tf.random.truncated_normal([self.filter_height_2, self.filter_width,
                                        self.output_channels, self.output_channels_2],
                                        dtype = tf.float32, stddev = 0.1),
                                        name = "filters")
        self.cnn_bias_2 = tf.Variable(tf.random.normal([self.output_channels_2]))

        self.pool_size_2 = 2
        self.pool_strides_2 = [2, 2]

        linear_input = int(32 / self.pool_strides[0] / self.pool_strides_2[0]
                      * 32 / self.pool_strides[1] / self.pool_strides_2[1]
                      * self.output_channels_2)
        """

        linear_input = int(32 / self.pool_strides[0]
                      * 32 / self.pool_strides[1] 
                      * self.output_channels)

        layer_1_output = 256
        output = 2

        self.W1 = tf.Variable(
            tf.random.truncated_normal([linear_input, layer_1_output],
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
                                    # self.filters_2, self.cnn_bias_2,
                                    self.W1, self.B1, self.W2, self.B2]
        self.optimal_variables = [self.filters, self.cnn_bias,
                                  # self.filters_2, self.cnn_bias_2,
                                  self.W1, self.B1, self.W2, self.B2]

    def use_optimal_variable(self):
        self.filters = self.optimal_variables[0]
        self.cnn_bias = self.optimal_variables[1]
        self.W1 = self.optimal_variables[2]
        self.B1 = self.optimal_variables[3]
        self.W2 = self.optimal_variables[4]
        self.B2 = self.optimal_variables[5]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # shape of input  (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of output (num_inputs, output_height, output_width, output_channels)
        c1_conv = tf.nn.conv2d(inputs, self.filters, self.strides, padding="SAME")
        c1_conv_bias = tf.nn.bias_add(c1_conv, self.cnn_bias)
        c1_relu = tf.nn.relu(c1_conv_bias)

        # pool_1
        pool_1 = tf.nn.max_pool2d(c1_relu, ksize=(self.pool_size, self.pool_size), 
                         strides=self.pool_strides, padding='SAME')

        # cnn-2
        # c1_conv = tf.nn.conv2d(pool_1, self.filters_2, self.strides_2, padding="SAME")
        # c1_conv_bias = tf.nn.bias_add(c1_conv, self.cnn_bias_2)
        # c1_relu = tf.nn.relu(c1_conv_bias)

        # pool_2 = tf.nn.max_pool2d(c1_relu, ksize=(self.pool_size_2, self.pool_size_2), 
        #                  strides=self.pool_strides_2, padding='SAME')

        # this reshape "flattens" the image data
        x0_inputs = tf.reshape(pool_1, [pool_1.shape[0], -1])

        x1 = linear_unit(x0_inputs, self.W1, self.B1)
        # apply ReLU activation
        x1_relu = tf.nn.relu(x1)
        x2 = linear_unit(x1_relu, self.W2, self.B2)

        return x2
    
    def update_optimal_variables(self):
        self.optimal_variables = copy.deepcopy(self.trainable_variables)
