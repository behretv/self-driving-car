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
        self.__mu = 0
        self.__sigma = 0.1
        self.__cost = None
        self.__logits = None
        self.__optimizer = None

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
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__params.learning_rate)
        self.__optimizer = optimizer.minimize(loss_operation)

    def compute_cost(self):
        prediction = tf.equal(tf.argmax(self.__logits, 1), tf.argmax(self.__tf_one_hot_labels, 1))
        self.__cost = tf.reduce_mean(tf.cast(prediction, tf.float32))

    def generate_network(self):
        layer_1 = self.__convolutional_layer(layer=self.tf_features, out_depth=self.__params.conv_depth[0])
        layer_2 = self.__convolutional_layer(layer=layer_1, out_depth=self.__params.conv_depth[1])

        fc = flatten(layer_2)

        layer_3 = self.__fully_connected_layer(layer=fc, out_sz=self.__params.out_size[0])
        layer_3 = self.__regularization(layer_3, 'dropout')
        layer_4 = self.__fully_connected_layer(layer=layer_3, out_sz=self.__params.out_size[1])
        layer_4 = self.__regularization(layer_4, 'dropout')

        self.__logits = self.__output_layer(layer=layer_4)

    def __output_layer(self, layer):
        shape_weights = (int(layer.shape[1]), self.__data.n_labels)
        shape_bias = (1, self.__data.n_labels)
        out_weight = tf.Variable(tf.truncated_normal(shape=shape_weights, mean=self.__mu, stddev=self.__sigma))
        out_bias = tf.Variable(tf.zeros(shape=shape_bias))
        return tf.matmul(layer, out_weight) + out_bias

    def __fully_connected_layer(self, layer, out_sz):
        shape_weight = (int(layer.shape[1]), out_sz)
        shape_bias = (1, out_sz)
        weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.__mu, stddev=self.__sigma))
        bias = tf.Variable(tf.zeros(shape=shape_bias))

        layer = tf.matmul(layer, weight) + bias
        return tf.nn.relu(layer)

    def __convolutional_layer(self, layer, out_depth):
        strides = self.__params.conv_strides
        out_sz = np.ceil(float(int(layer.shape[1]) - self.__params.conv_fsize + 1) / float(strides[1]))

        shape_weight = (self.__params.conv_fsize,
                        self.__params.conv_fsize,
                        int(layer.shape[3]),
                        out_depth)
        shape_bias = (1, out_sz, out_sz, out_depth)

        weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.__mu, stddev=self.__sigma))
        bias = tf.Variable(tf.zeros(shape=shape_bias))

        conv_layer = tf.nn.conv2d(layer, weight, strides=strides, padding='VALID') + bias
        return tf.nn.relu(conv_layer)

    def __regularization(self, layer, regularization_type: str):
        if regularization_type == 'pool':
            return tf.nn.max_pool(layer,
                                  ksize=self.__params.pool_ksize,
                                  strides=self.__params.pool_strides,
                                  padding='VALID')
        if regularization_type == 'dropout':
            return tf.nn.dropout(layer, keep_prob=self.tf_keep_prob)
