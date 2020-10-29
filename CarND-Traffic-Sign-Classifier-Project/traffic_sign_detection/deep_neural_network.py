""" Class to parametrize and define a deep neural network """
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class DeepNeuralNetwork:

    def __init__(self, tf_features, tf_labels, data):
        self.tf_features = tf_features
        self.tf_labels = tf_labels
        self.tf_one_hot_labels = tf.one_hot(self.tf_labels, data.n_labels)
        self.logits = None
        self._optimizer = None
        self.learning_rate = 0.001

        mu = 0
        sigma = 0.1

        self.weights = [
            tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
            tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma)),
        ]

        self.biases = [
            tf.Variable(tf.zeros(shape=(1, 28, 28, 6))),
            tf.Variable(tf.zeros(shape=(1, 10, 10, 16))),
            tf.Variable(tf.zeros(shape=(1, 120))),
            tf.Variable(tf.zeros(shape=(1, 84))),
            tf.Variable(tf.zeros(shape=(1, 43))),
        ]

    def generate_optimizer(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_one_hot_labels, logits=self.logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss_operation)

    def compute_cost(self):
        prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.tf_one_hot_labels, 1))
        return tf.reduce_mean(tf.cast(prediction, tf.float32))

    def process(self):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each
        # layer

        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer_1 = tf.nn.conv2d(self.tf_features, self.weights[0], strides=strides, padding=padding) + self.biases[0]
        layer_1 = tf.nn.relu(layer_1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer_1 = tf.nn.max_pool(layer_1, k, strides, padding)

        # Layer 2: Convolutional. Output = 10x10x16.
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer_2 = tf.nn.conv2d(layer_1, self.weights[1], strides=strides, padding=padding) + self.biases[1]
        layer_2 = tf.nn.relu(layer_2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        k = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

        # Flatten. Input = 5x5x16. Output = 400.
        fc = flatten(layer_2)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        layer_3 = tf.matmul(fc, self.weights[2]) + self.biases[2]
        layer_3 = tf.nn.relu(layer_3)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer_4 = tf.matmul(layer_3, self.weights[3]) + self.biases[3]
        layer_4 = tf.nn.relu(layer_4)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        self.logits = tf.matmul(layer_4, self.weights[4]) + self.biases[4]
