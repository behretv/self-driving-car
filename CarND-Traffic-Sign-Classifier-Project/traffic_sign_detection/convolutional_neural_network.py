""" Class to parametrize and define a deep neural network """
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.data_handler import DataHandler


class ConvolutionalNeuralNetwork:

    def __init__(self, data: DataHandler, hyper: HyperParameterHandler):
        # Properties which can be assigned by the input
        self.__img_depth = data.image_depth
        self.__img_shape = data.image_shape
        self.tf_features = tf.placeholder(tf.float32, shape=data.image_shape)
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
        self.__filter_size = 5

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
        out_sz1 = np.ceil(float(self.__img_shape[1] - self.__filter_size + 1)/float(2))
        layer_1 = self.__convolutional_layer(tf_feature=self.tf_features,
                                             img_sz=self.__img_shape[1],
                                             in_depth=self.__img_depth,
                                             out_depth=6)

        # Layer 2: Convolutional. Output = 10x10x16.
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        layer_2 = self.__convolutional_layer(tf_feature=layer_1,
                                             img_sz=14,
                                             in_depth=6,
                                             out_depth=16)

        # Flatten. Input = 5x5x16. Output = 400.
        fc = flatten(layer_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        layer_3 = self.__fully_connected_layer(tf_feature=fc, in_sz=400, out_sz=120)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = self.__fully_connected_layer(tf_feature=layer_3, in_sz=120, out_sz=84)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        out_weight = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=self.mu, stddev=self.sigma))
        out_bias = tf.Variable(tf.zeros(shape=(1, 43)))
        self.__logits = tf.matmul(layer_4, out_weight) + out_bias

    def __fully_connected_layer(self, tf_feature, in_sz, out_sz):
        weight = tf.Variable(tf.truncated_normal(shape=(in_sz, out_sz), mean=self.mu, stddev=self.sigma))
        bias = tf.Variable(tf.zeros(shape=(1, out_sz)))

        layer = tf.matmul(tf_feature, weight) + bias
        layer = tf.nn.relu(layer)
        return tf.nn.dropout(layer, keep_prob=self.tf_keep_prob)

    def __convolutional_layer(self, tf_feature, img_sz, in_depth, out_depth):
        strides = [1, 1, 1, 1]
        out_sz = np.ceil(float(img_sz - self.__filter_size + 1) / float(strides[1]))

        weight = tf.Variable(tf.truncated_normal(shape=(self.__filter_size, self.__filter_size, in_depth, out_depth),
                                                 mean=self.mu, stddev=self.sigma))
        bias = tf.Variable(tf.zeros(shape=(1, out_sz, out_sz, out_depth)))

        conv_layer = tf.nn.conv2d(tf_feature, weight, strides=strides, padding='VALID') + bias
        conv_layer = tf.nn.relu(conv_layer)

        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        return tf.nn.max_pool(conv_layer, k, strides=strides, padding='VALID')
