""" Class to parametrize and define a deep neural network """
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.data_handler import DataHandler


class ConvolutionalNeuralNetwork:

    def __init__(self, data: DataHandler, hyper: HyperParameterHandler):
        # Properties which can be assigned by the input
        self.__img_shape = data.image_shape
        self.tf_features = tf.placeholder(tf.float32, shape=self.__img_shape)
        self.tf_labels = tf.placeholder(tf.int32, None)
        self.tf_keep_prob = tf.placeholder(tf.float32)
        self.__tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)
        self.__learning_rate = hyper.parameter.learning_rate

        # Properties which have to be computed
        self.__cost = None
        self.__logits = None
        self.__optimizer = None

        self.mu = 0
        self.sigma = 0.1

        # Dimensions per layer:
        # Input: 32x32x3
        # Layer 1 conv: 5x5x3x6  --> ceil(32 - 5 + 1)/1 = 28
        # Layer 1 pool: 5x5x6x16  --> 28/2 = 14
        # Layer 2 conv: 5x5x3x6  --> ceil(14 - 5 + 1)/1 = 10
        # Layer 2 pool: 5x5x6x16  --> 10/2 = 5
        self.__depth = [6, 16]

        # (height, width, input_depth, output_depth)
        self._weights = [
            tf.Variable(tf.truncated_normal(shape=(5, 5, self.__img_shape[3], self.__depth[0]), mean=self.mu, stddev=self.sigma)),
            tf.Variable(tf.truncated_normal(shape=(5, 5, self.__depth[0], self.__depth[1]), mean=self.mu, stddev=self.sigma)),
            tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma)),
            tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma)),
            tf.Variable(tf.truncated_normal(shape=(84, 43), mean=self.mu, stddev=self.sigma)),
        ]

        self._biases = [
            tf.Variable(tf.zeros(shape=(1, 28, 28, self.__depth[0]))),
            tf.Variable(tf.zeros(shape=(1, 10, 10, self.__depth[1]))),
            tf.Variable(tf.zeros(shape=(1, 120))),
            tf.Variable(tf.zeros(shape=(1, 84))),
            tf.Variable(tf.zeros(shape=(1, 43))),
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
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        layer_1 = self.__convolutional_layer(tf_feature=self.tf_features,
                                             in_sz=32,
                                             sz_filter=5,
                                             in_depth=3,
                                             out_depth=6)

        # Layer 2: Convolutional. Output = 10x10x16.
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        layer_2 = self.__convolutional_layer(tf_feature=layer_1,
                                             in_sz=14,
                                             sz_filter=5,
                                             in_depth=6,
                                             out_depth=16)

        # Flatten. Input = 5x5x16. Output = 400.
        fc = flatten(layer_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        layer_3 = tf.matmul(fc, self._weights[2]) + self._biases[2]
        layer_3 = tf.nn.relu(layer_3)
        layer_3 = tf.nn.dropout(layer_3, keep_prob=self.tf_keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = tf.matmul(layer_3, self._weights[3]) + self._biases[3]
        layer_4 = tf.nn.relu(layer_4)
        #layer_4 = tf.nn.dropout(layer_4, keep_prob=self.tf_keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        self.__logits = tf.matmul(layer_4, self._weights[4]) + self._biases[4]

    def __convolutional_layer(self, tf_feature, in_sz, sz_filter, in_depth, out_depth):
        strides = [1, 1, 1, 1]
        out_sz = np.ceil(float(in_sz - sz_filter + 1) / float(strides[1]))

        weight = tf.Variable(tf.truncated_normal(shape=(sz_filter, sz_filter, in_depth, out_depth), mean=self.mu, stddev=self.sigma))
        bias = tf.Variable(tf.zeros(shape=(1, out_sz, out_sz, out_depth)))

        conv_layer = tf.nn.conv2d(tf_feature, weight, strides=strides, padding='VALID') + bias
        #conv_layer = tf.nn.bias_add(conv_layer, bias)
        conv_layer = tf.nn.relu(conv_layer)

        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        return tf.nn.max_pool(conv_layer, k, strides=strides, padding='VALID')
