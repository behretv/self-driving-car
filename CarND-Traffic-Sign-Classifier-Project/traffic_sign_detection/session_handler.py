""" This files handles the sessions, to train and test a neural network """
import logging
import math

import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from traffic_sign_detection.data_handler import DataType


class SessionHandler:

    def __init__(self, data, cost, session_file):
        self.data = data
        self.batch_size = 128
        self.epochs = 10
        self.cost = cost
        self.file = session_file

    def train(self, optimizer, tf_feature, tf_label):
        feature_train, label_train = self.data.get_shuffled_data(DataType.TRAIN)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            batch_count = int(math.ceil(len(feature_train) / self.batch_size))

            logging.info("Training...")
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
                    batch_x = feature_train[batch_start:batch_end]
                    batch_y = label_train[batch_start:batch_end]
                    sess.run(optimizer, feed_dict={tf_feature: batch_x, tf_label: batch_y})

                accuracy = self.validate(tf_feature, tf_label)

            saver.save(sess, self.file)
            print("Model saved")
        return accuracy

    def validate(self, tf_features, tf_labels):
        feature_valid, label_valid = self.data.get_shuffled_data(DataType.VALID)
        n_features = len(feature_valid)

        total_accuracy = 0
        tmp_sess = tf.get_default_session()
        for i_start in range(0, n_features, self.batch_size):
            i_end = i_start + self.batch_size
            tmp_features = feature_valid[i_start:i_end]
            tmp_labels = label_valid[i_start:i_end]
            tmp_accuracy = tmp_sess.run(self.cost, feed_dict={tf_features: tmp_features, tf_labels: tmp_labels})
            total_accuracy += (tmp_accuracy * len(tmp_features))
        return total_accuracy / n_features

    def test(self, tf_feature, tf_label):
        # Runs saved session
        saver = tf.train.Saver()
        feature_test, label_test = self.data.get_shuffled_data(DataType.TEST)
        with tf.Session() as sess:
            saver.restore(sess, self.file)
            test_accuracy = sess.run(self.cost, feed_dict={tf_feature: feature_test, tf_label: label_test})

        return test_accuracy
