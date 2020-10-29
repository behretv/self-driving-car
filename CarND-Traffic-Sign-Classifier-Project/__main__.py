import logging

from traffic_sign_detection.data_handler import DataHandler
from traffic_sign_detection.deep_neural_network import DeepNeuralNetwork
from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.session_handler import SessionHandler


def main():
    logging.basicConfig(level=logging.INFO)

    # 1 Data and hyper-parameters
    hyper_parameters = HyperParameterHandler()
    files = FileHandler()
    data = DataHandler(files)

    # 2 DNN
    deep_network = DeepNeuralNetwork(data, hyper_parameters)
    deep_network.generate_network()
    deep_network.generate_optimizer()
    deep_network.compute_cost()

    # 3 Run and save session
    session = SessionHandler(files, data, deep_network, hyper_parameters)
    valid_accuracy = session.train()
    test_accuracy = session.test()
    print("Valid Accuracy = {:.3f}".format(valid_accuracy))
    print('Test Accuracy = {:.3f}'.format(test_accuracy))


if __name__ == "__main__":
    main()
