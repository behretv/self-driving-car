""" This files handles the sessions, to train and tests a neural network """
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from tqdm import tqdm
from src.traffic_sign_detection.data_handler import DataType, DataHandler
from src.traffic_sign_detection.file_handler import FileHandler


class SessionHandler:

    def __init__(self, files: FileHandler, data: DataHandler):
        self.__logger = logging.getLogger(SessionHandler.__name__)
        self.__data = data
        self.__file = files.model_file
        self.__cnn = None  # cnn
        self.__params = None  # hyper.parameter
        self.__session = None
        self.__saver = None
        self.list_total_valid_accuracy = []
        self.list_train_accuracy = []
        self.list_valid_accuracy = []
        self.list_loss = []
        self.list_batch = []
        self.log_batch_step = 500

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

    @property
    def file(self):
        assert self.__file is not None
        self.__logger.info("Restore model: {}".format(self.__file))
        return self.__file

    @property
    def session(self):
        assert self.__session is not None
        return self.__session

    @property
    def valid_accuracy(self):
        assert len(self.list_total_valid_accuracy) > 0
        return self.list_total_valid_accuracy[-1]

    @session.setter
    def session(self, value):
        assert value is not None
        self.__logger.info("+++Session started!+++")
        self.__session = value

    def close(self):
        self.session.close()
        self.__logger.info("+++Session closed!+++")

    def __init_session(self):
        self.__saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.list_total_valid_accuracy = []
        self.list_train_accuracy = []
        self.list_valid_accuracy = []
        self.list_loss = []
        self.list_batch = []

    def __extract_batch_ranger(self, i):
        start = i * self.params.batch_size
        return start, start + self.params.batch_size

    def train_only(self):
        feature_train, label_train = self.__data.shuffled_data(DataType.TRAIN)
        feature_valid, label_valid = self.__data.shuffled_data(DataType.VALID)
        n_train_batches = self.__data.number_of_batches(DataType.TRAIN, self.params.batch_size)
        n_valid_batches = self.__data.number_of_batches(DataType.VALID, self.params.batch_size)

        self.__init_session()

        for i in range(self.params.epochs):
            # Progress bar
            batches_pbar = tqdm(range(n_train_batches), desc=self.__progress_msg(i), unit='batches')
            feature_train, label_train = shuffle(feature_train, label_train)

            for batch_i in batches_pbar:
                batch_range = self.__extract_batch_ranger(batch_i)
                train_batch = self.__generate_feed_dict(feature_train, label_train, batch_range, self.params.drop_out)
                self.__session.run(self.cnn.optimizer, feed_dict=train_batch)

            tmp_valid_accuracy = self.accuracy_running(feature_valid, label_valid, n_valid_batches)
            self.list_total_valid_accuracy.append(tmp_valid_accuracy)

            if not self.__is_accuracy_improved():
                break

    def train(self):
        feature_train, label_train = self.__data.shuffled_data(DataType.TRAIN)
        feature_valid, label_valid = self.__data.shuffled_data(DataType.VALID)
        n_train_batches = self.__data.number_of_batches(DataType.TRAIN, self.params.batch_size)
        n_valid_batches = self.__data.number_of_batches(DataType.VALID, self.params.batch_size)

        self.__init_session()

        for i in range(self.params.epochs):
            # Progress bar

            batches_pbar = tqdm(range(n_train_batches), desc=self.__progress_msg(i), unit='batches')

            feature_train, label_train = shuffle(feature_train, label_train)

            for batch_i in batches_pbar:
                batch_range = self.__extract_batch_ranger(batch_i)
                train_batch = self.__generate_feed_dict(feature_train, label_train, batch_range, 0.5)
                loss_batch = self.__generate_feed_dict(feature_train, label_train, batch_range, 1.0)
                valid_feed = self.__generate_feed_dict(feature_valid, label_valid, batch_range, 1.0)

                self.__session.run(self.cnn.optimizer, feed_dict=train_batch)

                # Log every 50 batches
                if not batch_i % self.log_batch_step:
                    # Calculate Training and Validation accuracy
                    loss = self.__session.run(self.cnn.cost, feed_dict=loss_batch)
                    train_accuracy = self.session.run(self.cnn.accuracy, feed_dict=loss_batch)
                    valid_accuracy = self.session.run(self.cnn.accuracy, feed_dict=valid_feed)

                    previous_batch = self.list_batch[-1] if self.list_batch else 0
                    self.list_batch.append(self.log_batch_step + previous_batch)
                    self.list_train_accuracy.append(train_accuracy)
                    self.list_valid_accuracy.append(valid_accuracy)
                    self.list_loss.append(loss)

            self.list_total_valid_accuracy.append(self.accuracy_running(feature_valid, label_valid, n_valid_batches))
            if not self.__is_accuracy_improved():
                break

    def save_session(self):
        self.__logger.info("Save model:")
        self.__saver.save(self.session, self.file)

    def accuracy_running(self, features, labels, number_of_batches):
        n_features = len(features)

        total_accuracy = 0
        for i_start in range(number_of_batches):
            batch_range = self.__extract_batch_ranger(i_start)
            valid_batch = self.__generate_feed_dict(features, labels, batch_range)
            tmp_accuracy = self.session.run(self.cnn.accuracy, feed_dict=valid_batch)

            total_accuracy += (tmp_accuracy * self.params.batch_size)
        return total_accuracy / n_features

    def accuracy_restored(self, datatype: DataType):
        features, labels = self.__data.shuffled_data(datatype)

        # Runs saved session
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.file)
            feed_dict = self.__generate_feed_dict(features, labels)
            return sess.run(self.cnn.accuracy, feed_dict=feed_dict)

    def prediction_restored(self, datatype: DataType):
        features, labels = self.__data.shuffled_data(datatype)

        # Runs saved session
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.file)
            feed_dict = self.__generate_feed_dict(features, labels)
            prediction = sess.run(self.cnn.prediction, feed_dict=feed_dict)
        return {'labels': labels, 'prediction': prediction}

    def softmax_restored(self, datatype: DataType):
        features, labels = self.__data.shuffled_data(datatype)

        # Runs saved session
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.file)
            feed_dict = self.__generate_feed_dict(features, labels)
            softmax = sess.run(self.cnn.softmax, feed_dict)
            top3 = sess.run(tf.nn.top_k(tf.constant(softmax), k=3))
        return top3, labels

    def visualize_training_process(self):
        batches = self.list_batch
        loss_batch = self.list_loss
        train_acc_batch = self.list_train_accuracy
        valid_acc_batch = self.list_valid_accuracy

        loss_plot = plt.subplot(211)
        loss_plot.set_title('Loss')
        loss_plot.plot(batches, loss_batch, 'g')
        loss_plot.set_xlim([batches[0], batches[-1]])
        acc_plot = plt.subplot(212)
        acc_plot.set_title('Accuracy')
        acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
        acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
        acc_plot.set_ylim([0, 1.0])
        acc_plot.set_xlim([batches[0], batches[-1]])
        acc_plot.legend(loc=4)
        plt.tight_layout()
        plt.show()

    def __generate_feed_dict(self, feature, label, idx_range=None, keep_prob=1.0):
        if idx_range is not None:
            feature = feature[idx_range[0]:idx_range[1]]
            label = label[idx_range[0]:idx_range[1]]
        return {
            self.cnn.tf_features: feature,
            self.cnn.tf_labels: label,
            self.cnn.tf_keep_prob: keep_prob
        }

    def __progress_msg(self, i):
        list_accuracy = self.list_total_valid_accuracy
        if i > 0:
            msg = 'Previous Accuracy={:.3f} Epoch {:>2}/{}'.format(list_accuracy[i - 1], i + 1, self.params.epochs)
        else:
            msg = 'Epoch {:>2}/{}'.format(i + 1, self.params.epochs)
        return msg

    def __is_accuracy_improved(self):
        list_accuracy = np.array(self.list_total_valid_accuracy)
        is_improved = True
        if len(list_accuracy) > 3:
            mean_diff = np.mean(np.diff(list_accuracy[-4:]))
            if mean_diff < 0.001:
                self.__logger.info("Abort, accuracy did not increase enough!")
                is_improved = False
        return is_improved
