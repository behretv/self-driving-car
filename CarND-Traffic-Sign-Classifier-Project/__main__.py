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
    session_handler = SessionHandler(files, data)
    covnet = ConvolutionalNeuralNetwork(data, hyper)

    for i in range(0, 1):
        logger.info("%d # ITERATION \n\n", i)

        # Generate new random parameter set
        if i > 0:
            hyper.next_parameter_set()

        # 2 DNN
        covnet.params = hyper.parameter
        covnet.generate_network()
        covnet.generate_cost()
        covnet.generate_optimizer()
        covnet.generate_accuracy()

        # 3 Run and save session
        session_handler.cnn = covnet
        session_handler.params = hyper.parameter
        logger.info('='*30)
        valid_accuracy = session_handler.train()
        logger.info("Valid Accuracy = {:.3f}".format(valid_accuracy))
        session_handler.visualize_training_process()

        # Check accuracy and update parameter file if increased
        hyper.update_accuracy(valid_accuracy, data.sample_size(DataType.TEST))
        if hyper.is_accuracy_improved:
            hyper.update_file()
            session_handler.save_session()

            # Check test accuracy
            logger.info('=' * 30)
            test_accuracy = session_handler.accuracy_restored(DataType.TEST)
            logger.info('Test Accuracy = {:.3f}'.format(test_accuracy))

            # Check internet accuracy
            logger.info('=' * 30)
            internet_accuracy = session_handler.accuracy_restored(DataType.INTERNET)
            logger.info('Internet Accuracy = {:.3f}'.format(internet_accuracy))
            internet_prediction = session_handler.prediction_restored(DataType.INTERNET)
            logger.info('Internet Accuracy = {}'.format(internet_prediction))

            internet_softmax = session_handler.softmax_restored(DataType.INTERNET)
            print(internet_softmax)

        else:
            logger.info("Keep: {} since accuracy did not improve!".format(files.hyper_parameter))
        session_handler.close()


if __name__ == "__main__":
    main()
