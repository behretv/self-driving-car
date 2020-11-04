""" This files handles the sessions, to train and test a neural network """
import logging
import math

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from traffic_sign_detection.data_handler import DataType, DataHandler
from traffic_sign_detection.file_handler import FileHandler


class SessionHandler:

    def __init__(self, files: FileHandler, data: DataHandler):
        self.__data = data
        self.__cnn = None  # cnn
        self.__params = None  # hyper.parameter
        self.__file = files.model_file
        self.__logger = logging.getLogger(SessionHandler.__name__)
        self.__session = None
        self.__saver = None

    @property
    def cnn(self):
        assert self.__cnn is not None
        return self.__cnn

    @cnn.setter
    def cnn(self, value):
        assert value is not None
        self.__cnn = value

    @property
    def params(self):
        assert self.__params is not None
        return self.__params

    @params.setter
    def params(self, value):
        assert value is not None
        self.__params = value

    def train(self, step):
        feature_train, label_train = self.__data.get_shuffled_data(DataType.TRAIN)
        batch_count = int(math.ceil(len(feature_train) / self.params.batch_size))

        self.__saver = tf.train.Saver()
        list_accuracy = np.array([])
        list_loss = []
        with tf.Session() as self.__session:
            self.__logger.info("{}# Training...".format(step))
            self.__session.run(tf.global_variables_initializer())

            for i in range(self.params.epochs):
                # Progress bar

                batches_pbar = tqdm(range(batch_count), desc=self.progress_msg(i, list_accuracy), unit='batches')

                feature_train, label_train = shuffle(feature_train, label_train)

                for batch_i in batches_pbar:
                    batch_start = batch_i * self.params.batch_size
                    batch_end = batch_start + self.params.batch_size

                    self.__session.run(self.cnn.optimizer,
                                       feed_dict={
                                           self.cnn.tf_features: feature_train[batch_start:batch_end],
                                           self.cnn.tf_labels: label_train[batch_start:batch_end],
                                           self.cnn.tf_keep_prob: 0.5
                                       })

                    loss = self.__session.run(self.cnn.cost,
                                              feed_dict={
                                                  self.cnn.tf_features: feature_train[batch_start:batch_end],
                                                  self.cnn.tf_labels: label_train[batch_start:batch_end],
                                                  self.cnn.tf_keep_prob: 1.0
                                              })
                    list_loss.append(loss)
                list_accuracy = np.append(list_accuracy, self.validate())
                if not self.is_accuracy_improved(list_accuracy):
                    break

            self.save_model(step)

        return list_accuracy[-1]

    def progress_msg(self, i, list_accuracy):
        if i > 0:
            msg = 'Previous Accuracy={:.3f} Epoch {:>2}/{}'.format(list_accuracy[i - 1], i + 1, self.params.epochs)
        else:
            msg = 'Epoch {:>2}/{}'.format(i + 1, self.params.epochs)
        return msg

    def is_accuracy_improved(self, list_accuracy):
        is_improved = True
        if len(list_accuracy) > 3:
            mean_diff = np.mean(np.diff(list_accuracy[-4:]))
            if mean_diff < 0.005:
                self.__logger.info("Abort, accuracy did not increase enough!")
                is_improved = False
        return is_improved

    def save_model(self, step):
        filename = self.__file + str(step)
        self.__logger.info("Save model as: {}".format(filename))
        self.__saver.save(self.__session, filename)

    def validate(self):
        feature_valid, label_valid = self.__data.get_shuffled_data(DataType.VALID)
        n_features = len(feature_valid)

        total_accuracy = 0
        tmp_sess = tf.get_default_session()
        for i_start in range(0, n_features, self.params.batch_size):
            i_end = i_start + self.params.batch_size
            tmp_features = feature_valid[i_start:i_end]
            tmp_labels = label_valid[i_start:i_end]

            tmp_accuracy = tmp_sess.run(self.cnn.accuracy, feed_dict={
                self.cnn.tf_features: tmp_features,
                self.cnn.tf_labels: tmp_labels,
                self.cnn.tf_keep_prob: 1.0
            })

            total_accuracy += (tmp_accuracy * len(tmp_features))
        return total_accuracy / n_features

    def test(self, step):
        filename = self.__file + str(step)
        self.__logger.info("Restore model: {}".format(filename))

        # Runs saved session
        saver = tf.train.Saver()
        feature_test, label_test = self.__data.get_shuffled_data(DataType.TEST)
        with tf.Session() as sess:
            saver.restore(sess, filename)
            test_accuracy = sess.run(self.cnn.accuracy, feed_dict={
                self.cnn.tf_features: feature_test,
                self.cnn.tf_labels: label_test,
                self.cnn.tf_keep_prob: 1.0
            })

        return test_accuracy
