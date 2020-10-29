""" Class to parametrize and define a deep neural network """
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler


class DeepNeuralNetwork:

    def __init__(self, data, hyper_params: HyperParameterHandler):
        # Properties which can be assigned by the input
        self.tf_features = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.tf_labels = tf.placeholder(tf.int32, None)
        self._tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)
        self._learning_rate = hyper_params.learning_rate

        # Properties which have to be computed
        self._cost = None
        self._logits = None
        self._optimizer = None

        mu = 0
        sigma = 0.1

        self._weights = [
            tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma)),
        ]

        self._biases = [
            tf.Variable(tf.zeros(shape=(1, 28, 28, 6))),
            tf.Variable(tf.zeros(shape=(1, 10, 10, 16))),
            tf.Variable(tf.zeros(shape=(1, 120))),
            tf.Variable(tf.zeros(shape=(1, 84))),
            tf.Variable(tf.zeros(shape=(1, 43))),
        ]

    @property
    def optimizer(self):
        assert self._optimizer is not None
        return self._optimizer

    @property
    def cost(self):
        assert self._cost is not None
        return self._cost

    def generate_optimizer(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._tf_one_hot_labels, logits=self._logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._optimizer = optimizer.minimize(loss_operation)

    def compute_cost(self):
        prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._tf_one_hot_labels, 1))
        self._cost = tf.reduce_mean(tf.cast(prediction, tf.float32))

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
        layer_1 = tf.nn.max_pool(layer_1, k, strides, padding)

        # Layer 2: Convolutional. Output = 10x10x16.
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer_2 = tf.nn.conv2d(layer_1, self._weights[1], strides=strides, padding=padding) + self._biases[1]
        layer_2 = tf.nn.relu(layer_2)

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

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = tf.matmul(layer_3, self._weights[3]) + self._biases[3]
        layer_4 = tf.nn.relu(layer_4)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        self._logits = tf.matmul(layer_4, self._weights[4]) + self._biases[4]
