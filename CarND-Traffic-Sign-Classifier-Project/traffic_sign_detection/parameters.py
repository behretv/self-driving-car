""" Class to collect all parameters """
import random
import logging


class Parameters:

    def __init__(self, tmp_dict: dict):
        self.__logger = logging.getLogger(Parameters.__name__)
        self.learning_rate = tmp_dict['learning_rate']
        self.epochs = tmp_dict['epochs']
        self.batch_size = tmp_dict['batch_size']
        self.accuracy = tmp_dict['accuracy']
        self.conv_fsize = 5
        self.conv_depth = [6, 16]
        self.conv_strides = [1, 1, 1, 1]
        self.pool_strides = [1, 2, 2, 1]
        self.pool_ksize = [1, 2, 2, 1]
        self.out_size = [120, 84]
        self.mu = 0
        self.sigma = 0.1

    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'accuracy': self.accuracy
        }

    def add_random_offset(self):
        self.learning_rate *= 10.0 / random.randint(5, 15)
        self.epochs += random.randint(-1, 4)
        self.batch_size += random.randint(-15, 15)

    def print(self):
        self.__logger.info("\n\t- accuracy=%.2f"
                           "\n\t- learning_rate=%.4f"
                           "\n\t- epochs=%d"
                           "\n\t- batch_size=%d"
                           ,
                           self.accuracy,
                           self.learning_rate,
                           self.epochs,
                           self.batch_size)
