""" This files handles the sessions, to train and test a neural network """
import logging
import math

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from traffic_sign_detection.data_handler import DataType, DataHandler
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.deep_neural_network import DeepNeuralNetwork


class SessionHandler:

    def __init__(self,
                 files: FileHandler,
                 data: DataHandler,
                 dnn: DeepNeuralNetwork,
                 hyper: HyperParameterHandler):

        self.data = data
        self.dnn = dnn
        self.batch_size = hyper.parameter.batch_size
        self.epochs = hyper.parameter.epochs
        self.file = files.model_file
        self._logger = logging.getLogger(SessionHandler.__name__)

    def train(self):
        feature_train, label_train = self.data.get_shuffled_data(DataType.TRAIN)
        batch_count = int(math.ceil(len(feature_train) / self.batch_size))

        saver = tf.train.Saver()
        list_accuracy = np.array([])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self._logger.info("Training...")
            accuracy = 0.0
            for i in range(self.epochs):

                # Progress bar
                batches_pbar = tqdm(range(batch_count),
                                    desc='Previous Accuracy={:.3f} Epoch {:>2}/{}'.format(accuracy, i + 1, self.epochs),
                                    unit='batches')

                feature_train, label_train = shuffle(feature_train, label_train)
                for batch_i in batches_pbar:
                    batch_start = batch_i * self.batch_size
                    batch_end = batch_start + self.batch_size
                    sess.run(self.dnn.optimizer, feed_dict={
                        self.dnn.tf_features: feature_train[batch_start:batch_end],
                        self.dnn.tf_labels: label_train[batch_start:batch_end],
                        self.dnn.tf_keep_prob: 0.5
                    })

                accuracy = self.validate()
                list_accuracy = np.append(list_accuracy, accuracy)

                if len(list_accuracy) > 3:
                    mean_diff = np.mean(np.diff(list_accuracy[-3:]))
                    if mean_diff < 0.01:
                        self._logger.info("Abort, accuracy did not increase enough!")
                        break

            saver.save(sess, self.file)
            print("Model saved")
        return accuracy

    def validate(self):
        feature_valid, label_valid = self.data.get_shuffled_data(DataType.VALID)
        n_features = len(feature_valid)

        total_accuracy = 0
        tmp_sess = tf.get_default_session()
        for i_start in range(0, n_features, self.batch_size):
            i_end = i_start + self.batch_size
            tmp_features = feature_valid[i_start:i_end]
            tmp_labels = label_valid[i_start:i_end]
            tmp_accuracy = tmp_sess.run(self.dnn.cost, feed_dict={
                self.dnn.tf_features: tmp_features,
                self.dnn.tf_labels: tmp_labels,
                self.dnn.tf_keep_prob: 1.0
            })
            total_accuracy += (tmp_accuracy * len(tmp_features))
        return total_accuracy / n_features

    def test(self):
        # Runs saved session
        saver = tf.train.Saver()
        feature_test, label_test = self.data.get_shuffled_data(DataType.TEST)
        with tf.Session() as sess:
            saver.restore(sess, self.file)
            test_accuracy = sess.run(self.dnn.cost, feed_dict={
                self.dnn.tf_features: feature_test,
                self.dnn.tf_labels: label_test,
                self.dnn.tf_keep_prob: 1.0
            })

        return test_accuracy
