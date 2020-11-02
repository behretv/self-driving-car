""" This files handles the sessions, to train and test a neural network """
import logging
import math

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

    def train(self):
        feature_train, label_train = self.data.get_shuffled_data(DataType.TRAIN)
        batch_count = int(math.ceil(len(feature_train) / self.batch_size))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

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
                    sess.run(self.dnn.optimizer, feed_dict={
                        self.dnn.tf_features: batch_x,
                        self.dnn.tf_labels: batch_y,
                        self.dnn.tf_keep_prob: 0.5
                    })

                    '''
                    # Log every 50 batches
                    if not batch_i % log_batch_step:
                        # Calculate Training and Validation accuracy
                        training_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)
                        validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)

                        # Log batches
                        previous_batch = batches[-1] if batches else 0
                        batches.append(log_batch_step + previous_batch)
                        loss_batch.append(l)
                        train_acc_batch.append(training_accuracy)
                        valid_acc_batch.append(validation_accuracy)
                    '''

                accuracy = self.validate()

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
