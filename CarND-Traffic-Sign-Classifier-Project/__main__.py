import logging

import tensorflow as tf

from traffic_sign_detection.data_handler import DataHandler
from traffic_sign_detection.deep_neural_network import DeepNeuralNetwork
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.session_handler import SessionHandler
from traffic_sign_detection.file_handler import FileHandler


def main():
    logging.basicConfig(level=logging.INFO)

    files = FileHandler()
    hyper_parameters = HyperParameterHandler()
    data = DataHandler(files)

    # 2 Placeholders
    tf_feature = tf.placeholder(tf.float32, (None, 32, 32, 3))
    tf_label = tf.placeholder(tf.int32, None)

    # 3 DNN
    deep_network = DeepNeuralNetwork(tf_feature, tf_label, data)
    deep_network.process()
    optimizer = deep_network.generate_optimizer()
    cost = deep_network.compute_cost()

    # 5 Run and save session
    session = SessionHandler(files, data, cost, hyper_parameters)
    valid_accuracy = session.train(optimizer, tf_feature, tf_label)
    test_accuracy = session.test(tf_feature, tf_label)
    print("Valid Accuracy = {:.3f}".format(valid_accuracy))
    print('Test Accuracy = {:.3f}'.format(test_accuracy))


if __name__ == "__main__":
    main()
