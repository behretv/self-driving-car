""" Class to parametrize and define a deep neural network """
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.data_handler import DataHandler


class ConvolutionalNeuralNetwork:

    def __init__(self, data: DataHandler, hyper: HyperParameterHandler):
        # Properties which can be assigned by the input
        self.__data = data
        self.__learning_rate = hyper.parameter.learning_rate

        # Properties which have to be computed
        self.__cost = None
        self.__logits = None
        self.__optimizer = None

        self.__mu = 0
        self.__sigma = 0.1
        self.__depth = [6, 16]
        self.__filter_size = 5

        self.tf_features = tf.placeholder(tf.float32, shape=data.image_shape)
        self.tf_labels = tf.placeholder(tf.int32, None)
        self.tf_keep_prob = tf.placeholder(tf.float32)
        self.__tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)

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
        layer_1 = self.__convolutional_layer(layer=self.tf_features, out_depth=6)

        # Layer 2: Convolutional. Output = 10x10x16.
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        layer_2 = self.__convolutional_layer(layer=layer_1, out_depth=16)

        # Flatten. Input = 5x5x16. Output = 400.
        fc = flatten(layer_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        layer_3 = self.__fully_connected_layer(layer=fc, out_sz=120)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = self.__fully_connected_layer(layer=layer_3, out_sz=84)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        self.__logits = self.__output_layer(layer=layer_4)

    def __output_layer(self, layer):
        shape = (int(layer.shape[1]), self.__data.n_labels)
        out_weight = tf.Variable(tf.truncated_normal(shape=shape, mean=self.__mu, stddev=self.__sigma))
        out_bias = tf.Variable(tf.zeros(shape=(1, shape[1])))
        return tf.matmul(layer, out_weight) + out_bias

    def __fully_connected_layer(self, layer, out_sz):
        shape = (int(layer.shape[1]), out_sz)
        weight = tf.Variable(tf.truncated_normal(shape=shape, mean=self.__mu, stddev=self.__sigma))
        bias = tf.Variable(tf.zeros(shape=(1, out_sz)))

        layer = tf.matmul(layer, weight) + bias
        layer = tf.nn.relu(layer)
        return tf.nn.dropout(layer, keep_prob=self.tf_keep_prob)

    def __convolutional_layer(self, layer, out_depth):
        strides = [1, 1, 1, 1]
        img_sz = int(layer.shape[1])
        img_depth = int(layer.shape[3])
        out_sz = np.ceil(float(img_sz - self.__filter_size + 1) / float(strides[1]))

        shape_weight = (self.__filter_size, self.__filter_size, img_depth, out_depth)
        shape_bias = (1, out_sz, out_sz, out_depth)

        weight = tf.Variable(tf.truncated_normal(shape=shape_weight, mean=self.__mu, stddev=self.__sigma))
        bias = tf.Variable(tf.zeros(shape=shape_bias))

        conv_layer = tf.nn.conv2d(layer, weight, strides=strides, padding='VALID') + bias
        conv_layer = tf.nn.relu(conv_layer)

        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        return tf.nn.max_pool(conv_layer, k, strides=strides, padding='VALID')
