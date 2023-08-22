from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import sys
import tensorflow as tf
import numpy as np
import random


def linear_unit(x, W, b):
  return tf.matmul(x, W) + b

class ModelPart0:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")


        self.trainable_variables = [self.W1, self.B1]


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
        x = linear_unit(inputs, self.W1, self.B1)

        return x

def loss(logits, labels):
    """
    Calculates the cross-entropy loss after one forward pass.
    :param logits: during training, a matrix of shape (batch_size, self.num_classes)
    containing the result of multiple convolution and feed forward layers
    Softmax is applied in this function.
    :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
    :return: the loss of the model as a Tensor
    """
    cross_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    return np.mean(cross_loss)

def accuracy(logits, labels):
    """
    Calculates the model's prediction accuracy by comparing
    logits to correct labels – no need to modify this.
    :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

    NOTE: DO NOT EDIT

    :return: the accuracy of the model as a Tensor
    """
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    # Generate a shuffled index order
    num_elements = tf.shape(train_inputs)[0]
    shuffled_indices = tf.random.shuffle(tf.range(num_elements))

    shuffled_train_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_train_labels = tf.gather(train_labels, shuffled_indices)

    # Each epoch is an iteration over the training inputs and labels
    for start in range(0, num_elements, model.batch_size):
        inputs = shuffled_train_inputs[start:start + model.batch_size]
        labels = shuffled_train_labels[start:start + model.batch_size]

        # Implement back-prop:
        # For every batch, compute then descend the gradients for the model's weights
        with tf.GradientTape() as tape:
            predictions = model(inputs) # this calls the call function conveniently
            cross_loss = loss(predictions, labels)

        gradients = tape.gradient(cross_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """
    return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (10, num_classes)
    :param image_labels: the labels from get_data(), shape (10, num_classes)
    :param first_label: the name of the first class, "dog"
    :param second_label: the name of the second class, "cat"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
    plt.show()


CLASS_CAT = 3
CLASS_DOG = 5
EPOCHS = 25

def main(cifar10_data_folder):
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    25 epochs.
    You should receive a final accuracy on the testing examples for cat and dog
    of ~60% for Part1 and ~70% for Part3.
    :return: None
    '''
    train_inputs, train_labels = get_data(os.path.join(cifar10_data_folder, "train"),
                                          CLASS_CAT, CLASS_DOG)
    test_inputs, test_labels = get_data(os.path.join(cifar10_data_folder, "test"),
                                        CLASS_CAT, CLASS_DOG)
		
	# Create Model
    model = ModelPart0()
    
    metric = tf.keras.metrics.Accuracy()
    for epoch in range(0, EPOCHS):
        train(model, train_inputs, train_labels)
        predictions = model.call(train_inputs)
        metric.update_state(predictions, train_labels)
        train_acc = metric.result().numpy()
        c_loss = loss(predictions, train_labels)
        print(f"Training Accuracy: {train_acc}, loss: {c_loss} after {epoch + 1} epochs")

    # Train accuracy
    train_accuracy = model.accuracy(model.call(train_inputs), train_labels)
    print(f"Final train accuracy is: {train_accuracy}")

    # Test the accuracy by calling test() after running train()
    test_accuracy = test(model, test_inputs, test_labels)
    print(f"test accuracy is: {test_accuracy}")

    # # Visualize the data by using visualize_results()
    # random_nums = np.random.randint(1, 10000, size=10)
    # visualize_results(test_inputs[random_nums], model.call(test_inputs[random_nums]), test_labels[random_nums])

    print("end of assignment 1")

if __name__ == '__main__':
    # Check if any command-line arguments are provided
    if len(sys.argv) > 1:
        # use the first argument as the directory
        directory = sys.argv[1]
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(directory):
        cifar_data_folder = os.path.join(directory, 'CIFAR_data')
        main(cifar_data_folder)
    else:
        print(f"Directory {directory} does not exist!")


