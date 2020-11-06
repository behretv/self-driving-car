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
        self.conv_fsize = tmp_dict['cnn_filter_size']
        self.conv_depth = tmp_dict['cnn_depth']
        self.out_size = tmp_dict['out_size']
        self.sigma = tmp_dict['sigma']
        self.drop_out = tmp_dict['drop_out']

        test = tmp_dict['out_size']

        # Constant parameters
        self.mu = 0
        self.conv_strides = [1, 1, 1, 1]
        self.pool_strides = [1, 2, 2, 1]
        self.pool_ksize = [1, 2, 2, 1]

    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'cnn_filter_size': self.conv_fsize,
            'cnn_depth': self.conv_depth,
            'out_size': self.out_size,
            'sigma': self.sigma,
            'drop_out': self.drop_out,
            'accuracy': self.accuracy,
        }

    def add_random_offset(self):
        self.learning_rate *= 10.0 / random.randint(5, 15)
        self.epochs = random.randint(5, 10)
        self.batch_size = random.randint(100, 140)
        self.conv_fsize = random.randint(4, 6)
        self.sigma = 1.0/random.randint(9, 11)
        self.out_size = [random.randint(100, 140), random.randint(75, 90)]
        self.drop_out = random.uniform(0.5, 1.0)

    def print(self):
        str_dict = ""
        for entry in self.to_dict().items():
            str_dict += "\n\t- {}".format(entry)
        self.__logger.info(str_dict)
