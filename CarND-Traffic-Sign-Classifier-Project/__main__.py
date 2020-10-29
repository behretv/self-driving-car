# Load pickled data
import logging

import tensorflow as tf

from traffic_sign_detection.data_handler import DataHandler, DataType
from traffic_sign_detection.deep_neural_network import DeepNeuralNetwork
from traffic_sign_detection.session_handler import SessionHandler


def main():
    model_session_file = 'data/parameters/train_model.ckpt'
    training_file = 'data/input/train.p'
    validation_file = 'data/input/test.p'
    testing_file = 'data/input/valid.p'

    logging.basicConfig(level=logging.INFO)

    # 1 Data handling
    data = DataHandler(training_file, testing_file, validation_file)

    # 2 Placeholders
    tf_feature = tf.placeholder(tf.float32, (None, 32, 32, 3))
    tf_label = tf.placeholder(tf.int32, None)

    # 3 DNN
    deep_network = DeepNeuralNetwork(tf_feature, tf_label, data)
    deep_network.process()
    optimizer = deep_network.generate_optimizer()
    cost = deep_network.compute_cost()

    # 5 Run and save session
    session = SessionHandler(data, cost, model_session_file)
    valid_accuracy = session.train(optimizer, tf_feature, tf_label)
    test_accuracy = session.test(tf_feature, tf_label)
    print("Valid Accuracy = {:.3f}".format(valid_accuracy))
    print('Test Accuracy = {:.3f}'.format(test_accuracy))


if __name__ == "__main__":
    main()
