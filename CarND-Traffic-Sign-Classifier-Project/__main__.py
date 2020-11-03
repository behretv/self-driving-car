import logging

from traffic_sign_detection.data_handler import DataHandler, DataType
from traffic_sign_detection.convolutional_neural_network import ConvolutionalNeuralNetwork
from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.session_handler import SessionHandler


def main():
    logging.basicConfig(level=logging.INFO)

    # 1 Data and hyper-parameter
    files = FileHandler()
    data = DataHandler(files)
    #data.visualize_random_image()
    data.process()
    #data.visualize_labels_histogram()
    hyper = HyperParameterHandler(files)

    for i in range(0, 3):
        logging.info("%d # ITERATION \n\n", i)
        hyper.next_parameter_set()

        # 2 DNN
        covnet = ConvolutionalNeuralNetwork(data, hyper)
        covnet.generate_network()
        covnet.generate_optimizer()
        covnet.compute_cost()

        # 3 Run and save session
        session = SessionHandler(files, data, covnet, hyper)
        valid_accuracy, sess = session.train()
        test_accuracy = session.test()

        hyper.update(test_accuracy, data.sample_size(DataType.TEST))
        hyper.update_file_if_accuracy_improved()
        logging.info("Valid Accuracy = {:.3f}".format(valid_accuracy))
        logging.info('Test Accuracy = {:.3f}'.format(test_accuracy))


if __name__ == "__main__":
    main()
