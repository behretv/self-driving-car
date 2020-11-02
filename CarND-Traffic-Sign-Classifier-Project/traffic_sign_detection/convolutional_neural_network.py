""" Class to parametrize and define a deep neural network """
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler


class ConvolutionalNeuralNetwork:

    def __init__(self, data, hyper: HyperParameterHandler):
        # Properties which can be assigned by the input
        shape_features = (None, ) + data.n_features
        self.tf_features = tf.placeholder(tf.float32, shape=shape_features)
        self.tf_labels = tf.placeholder(tf.int32, None)
        self.tf_keep_prob = tf.placeholder(tf.float32)
        self.__tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)
        self.__learning_rate = hyper.parameter.learning_rate

        # Properties which have to be computed
        self.__cost = None
        self.__logits = None
        self.__optimizer = None

        mu = 0
        sigma = 0.1

        # Dimensions per layer:
        # Input: 32x32x3
        # Layer 1 conv: 5x5x3x6  --> ceil(32 - 5 + 1)/1 = 28
        # Layer 1 pool: 5x5x6x16  --> 28/2 = 14
        # Layer 2 conv: 5x5x3x6  --> ceil(14 - 5 + 1)/1 = 10
        # Layer 2 pool: 5x5x6x16  --> 10/2 = 5
        self.__depth = [3, 6, 16]

        # (height, width, input_depth, output_depth)
        self._weights = [
            tf.Variable(tf.truncated_normal(shape=(5, 5, self.__depth[0], self.__depth[1]), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(5, 5, self.__depth[1], self.__depth[2]), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma)),
        ]

        self._biases = [
            tf.Variable(tf.zeros(shape=(1, 28, 28, self.__depth[1]))),
            tf.Variable(tf.zeros(shape=(1, 10, 10, self.__depth[2]))),
            tf.Variable(tf.zeros(shape=(1, 120))),
            tf.Variable(tf.zeros(shape=(1, 84))),
            tf.Variable(tf.zeros(shape=(1, 43))),
        ]

        # (batch, height, width, depth)
        self.__strides = [
            [1, 1, 1, 1],
        ]

    @property
    def optimizer(self):
        assert self.__optimizer is not None
        return self.__optimizer

    @property
    def cost(self):
        assert self.__cost is not None
        return self.__cost

    def generate_optimizer(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.__tf_one_hot_labels, logits=self.__logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate)
        self.__optimizer = optimizer.minimize(loss_operation)

    def compute_cost(self):
        prediction = tf.equal(tf.argmax(self.__logits, 1), tf.argmax(self.__tf_one_hot_labels, 1))
        self.__cost = tf.reduce_mean(tf.cast(prediction, tf.float32))

    def generate_network(self):
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer_1 = tf.nn.conv2d(self.tf_features, self._weights[0], strides=strides, padding=padding) + self._biases[0]
        layer_1 = tf.nn.relu(layer_1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer_1 = tf.nn.max_pool(layer_1, k, strides=strides, padding=padding)

        # Layer 2: Convolutional. Output = 10x10x16.
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer_2 = tf.nn.conv2d(layer_1, self._weights[1], strides=strides, padding=padding) + self._biases[1]
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.tf_keep_prob)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

        # Flatten. Input = 5x5x16. Output = 400.
        fc = flatten(layer_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        layer_3 = tf.matmul(fc, self._weights[2]) + self._biases[2]
        layer_3 = tf.nn.relu(layer_3)
        #layer_3 = tf.nn.dropout(layer_3, keep_prob=self.tf_keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = tf.matmul(layer_3, self._weights[3]) + self._biases[3]
        layer_4 = tf.nn.relu(layer_4)
        #layer_4 = tf.nn.dropout(layer_4, keep_prob=self.tf_keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        self.__logits = tf.matmul(layer_4, self._weights[4]) + self._biases[4]

        def __compute_new_shape(height, width, filter_height, filter_width, strides):
            new_height = np.ceil(float(height - filter_height + 1) / float(strides[1]))
            new_width = np.ceil(float(width - filter_width + 1) / float(strides[2]))
            return new_height, new_width
