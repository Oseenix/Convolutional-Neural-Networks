import tensorflow as tf
import numpy as np
import copy

def linear_unit(x, W, b):
  return tf.matmul(x, W) + b

class ModelPart1:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        layer_1_output = 256
        output = 2

        self.W1 = tf.Variable(tf.random.truncated_normal([input, layer_1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, layer_1_output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.W2 = tf.Variable(tf.random.truncated_normal([layer_1_output, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

        self.trainable_variables = [self.W1, self.B1, self.W2, self.B2]
        self.optimal_variables = [self.W1, self.B1, self.W2, self.B2]

    def use_optimal_variable(self):
        self.W1 = self.optimal_variables[0]
        self.B1 = self.optimal_variables[1]
        self.W2 = self.optimal_variables[2]
        self.B2 = self.optimal_variables[3]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

        # this reshape "flattens" the image data
        inputs = np.reshape(inputs, [inputs.shape[0],-1])
        x1 = linear_unit(inputs, self.W1, self.B1)
        # apply ReLU activation
        x1_relu = tf.nn.relu(x1)
        x2 = linear_unit(x1_relu, self.W2, self.B2)

        return x2
    
    def update_optimal_variables(self):
        self.optimal_variables = copy.deepcopy(self.trainable_variables)