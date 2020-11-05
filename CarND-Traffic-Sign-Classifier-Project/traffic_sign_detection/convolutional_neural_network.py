""" Class to parametrize and define a deep neural network """
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.data_handler import DataHandler


class ConvolutionalNeuralNetwork:

    def __init__(self, data: DataHandler, hyper: HyperParameterHandler):
        # Public attributes
        self.tf_features = tf.placeholder(tf.float32, shape=data.image_shape)
        self.tf_labels = tf.placeholder(tf.int32, None)
        self.tf_keep_prob = tf.placeholder(tf.float32)

        # Private attributes
        self.__data = data
        self.__params = hyper.parameter
        self.__tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)
        self.__prediction = None
        self.__accuracy = None
        self.__logits = None
        self.__optimizer = None
        self.__cost = None

    @property
    def optimizer(self):
        assert self.__optimizer is not None
        return self.__optimizer

    @property
    def prediction(self):
        assert self.__prediction is not None
        return self.__prediction

    @property
    def accuracy(self):
        assert self.__accuracy is not None
        return self.__accuracy

    @property
    def cost(self):
        assert self.__cost is not None
        return self.__cost

    @property
    def params(self):
        assert self.__params is not None
        return self.__params

    @params.setter
    def params(self, value):
        assert value is not None
        self.__params = value

    @property
    def logits(self):
        assert self.__logits is not None
        return self.__logits

    def generate_cost(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.__tf_one_hot_labels, logits=self.__logits)
        self.__cost = tf.reduce_mean(cross_entropy)

    def generate_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
        self.__optimizer = optimizer.minimize(self.cost)

    def generate_accuracy(self):
        self.__prediction = tf.argmax(self.logits, 1)
        binary_predictions = tf.equal(self.prediction, tf.argmax(self.__tf_one_hot_labels, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(binary_predictions, tf.float32))

    def generate_network(self):
        layer_1 = self.__layer_convolution(layer=self.tf_features, out_depth=self.params.conv_depth[0])
        layer_2 = self.__layer_convolution(layer=layer_1, out_depth=self.params.conv_depth[1])

        fc = flatten(layer_2)

        layer_3 = self.__layer_fully_connected(layer=fc, out_sz=self.params.out_size[0])
        layer_3 = self.__regularization(layer_3, 'dropout')
        layer_4 = self.__layer_fully_connected(layer=layer_3, out_sz=self.params.out_size[1])
        layer_4 = self.__regularization(layer_4, 'dropout')

        self.__logits = self.__layer_output(layer=layer_4)

    def __layer_output(self, layer):
        shape_weight = (int(layer.shape[1]), self.__data.n_labels)
        shape_bias = (1, self.__data.n_labels)
        out_weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.params.mu, stddev=self.params.sigma))
        out_bias = tf.Variable(tf.zeros(shape=shape_bias))
        return tf.matmul(layer, out_weight) + out_bias

    def __layer_fully_connected(self, layer, out_sz):
        shape_weight = (int(layer.shape[1]), out_sz)
        shape_bias = (1, out_sz)
        weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.params.mu, stddev=self.params.sigma))
        bias = tf.Variable(tf.zeros(shape=shape_bias))

        layer = tf.matmul(layer, weight) + bias
        return tf.nn.relu(layer)

    def __layer_convolution(self, layer, out_depth):
        strides = self.params.conv_strides
        out_sz = np.ceil(float(int(layer.shape[1]) - self.params.conv_fsize + 1) / float(strides[1]))

        shape_weight = (self.params.conv_fsize,
                        self.params.conv_fsize,
                        int(layer.shape[3]),
                        out_depth)
        shape_bias = (1, out_sz, out_sz, out_depth)

        weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.params.mu, stddev=self.params.sigma))
        bias = tf.Variable(tf.zeros(shape=shape_bias))

        conv_layer = tf.nn.conv2d(layer, weight, strides=strides, padding='VALID') + bias
        return tf.nn.relu(conv_layer)

    def __regularization(self, layer, regularization_type: str):
        if regularization_type == 'pool':
            return tf.nn.max_pool(layer,
                                  ksize=self.params.pool_ksize,
                                  strides=self.params.pool_strides,
                                  padding='VALID')
        if regularization_type == 'dropout':
            return tf.nn.dropout(layer, keep_prob=self.tf_keep_prob)
