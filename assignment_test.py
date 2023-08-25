import unittest
import tensorflow as tf
from assignment import *
from convolution import *

class TestAssignment2(unittest.TestCase):
    def assertSequenceEqual(self, arr1, arr2):
        np.testing.assert_almost_equal(arr1, arr2, decimal=5, err_msg='', verbose=True)

    def test_loss(self):
        '''
        Simple test to make sure loss function is the average softmax cross-entropy loss

        NOTE: DO NOT EDIT
        '''
        labels = tf.constant([[1.0, 0.0]])
        logits = tf.constant([[1.0, 0.0]])
        self.assertAlmostEqual(np.mean(loss(logits, labels)), 0.31326166)
        logits = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(np.mean(loss(logits, labels)), 0.8132616281509399)

    def test_input_reshape(self):
        

if __name__ == '__main__':
    unittest.main()


