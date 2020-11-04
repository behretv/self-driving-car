import logging
import coloredlogs

from traffic_sign_detection.data_handler import DataHandler, DataType
from traffic_sign_detection.convolutional_neural_network import ConvolutionalNeuralNetwork
from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.session_handler import SessionHandler


def train():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='INFO', logger=logger)

    # 1 Data and hyper-parameter
    files = FileHandler()
    data = DataHandler(files)
    #data.visualize_random_image()
    data.process()
    #data.visualize_labels_histogram()
    hyper = HyperParameterHandler(files)
    session = SessionHandler(files, data)

    for i in range(0, 3):
        logger.info("%d # ITERATION \n\n", i)
        hyper.next_parameter_set()

        # 2 DNN
        covnet = ConvolutionalNeuralNetwork(data, hyper)
        covnet.generate_network()
        covnet.generate_optimizer()
        covnet.compute_cost()

        # 3 Run and save session
        session.cnn = covnet
        session.params = hyper.parameter
        valid_accuracy = session.train(i)
        logger.info("Valid Accuracy = {:.3f}".format(valid_accuracy))
        hyper.update(valid_accuracy, data.sample_size(DataType.TEST))
        hyper.update_file_if_accuracy_improved()

        test_accuracy = session.test(i)
        logger.info('Test Accuracy = {:.3f}'.format(test_accuracy))


if __name__ == "__main__":
    train()
