import logging
import coloredlogs

from traffic_sign_detection.data_handler import DataHandler, DataType
from traffic_sign_detection.convolutional_neural_network import ConvolutionalNeuralNetwork
from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler
from traffic_sign_detection.session_handler import SessionHandler


def main():
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

    for i in range(0, 1):
        logger.info("%d # ITERATION \n\n", i)

        # 2 DNN
        covnet = ConvolutionalNeuralNetwork(data, hyper)
        covnet.generate_network()
        covnet.generate_cost()
        covnet.generate_optimizer()
        covnet.generate_accuracy()

        # 3 Run and save session
        session.cnn = covnet
        session.params = hyper.parameter
        valid_accuracy = session.train(i)
        logger.info("Valid Accuracy = {:.3f}".format(valid_accuracy))
        session.visualize_training_process()

        hyper.update_accuracy(valid_accuracy, data.sample_size(DataType.TEST))
        if hyper.is_accuracy_improved:
            hyper.update_file()
        else:
            logger.info("Keep: {} since accuracy did not improve!".format(files.hyper_parameter))

        test_accuracy = session.test(i)
        logger.info('Test Accuracy = {:.3f}'.format(test_accuracy))

        # Generate new random parameter set
        hyper.next_parameter_set()


if __name__ == "__main__":
    main()
