""" Class to collect hyper parameter and optimize them """
import json
import logging
import random
from traffic_sign_detection.file_handler import FileHandler


class Parameters:

    def __init__(self, tmp_dict: dict):
        self.learning_rate = tmp_dict['learning_rate']
        self.epochs = tmp_dict['epochs']
        self.batch_size = tmp_dict['batch_size']
        self.accuracy = tmp_dict['accuracy']

    def to_dict(self):
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'accuracy': self.accuracy
        }

    def add_random_offset(self):
        self.learning_rate *= 10.0 / random.randint(5, 15)
        self.epochs += random.randint(-5, 2)
        self.batch_size += random.randint(-20, 20)


class HyperParameterHandler:
    def __init__(self, files: FileHandler):
        self.__logger = logging.getLogger(HyperParameterHandler.__name__)
        self.__file = files.hyper_parameter
        self.__parameter = Parameters(self.__read_file())
        self.__is_accuracy_improved = False

    @property
    def parameter(self):
        return self.__parameter

    @parameter.setter
    def parameter(self, value):
        assert isinstance(value, Parameters)
        self.__parameter = value

    @property
    def is_accuracy_improved(self):
        return self.__is_accuracy_improved

    def next_parameter_set(self):
        self.__parameter = Parameters(self.__read_file())
        self.__parameter.add_random_offset()
        self.print_parameters()

    def print_parameters(self):
        self.__logger.info("\n\t- accuracy=%.2f"
                           "\n\t- learning_rate=%.4f"
                           "\n\t- epochs=%d"
                           "\n\t- batch_size=%d"
                           ,
                           self.parameter.accuracy,
                           self.parameter.learning_rate,
                           self.parameter.epochs,
                           self.parameter.batch_size)

    def update(self, new_accuracy, sample_size):
        """ Update only if rule of 30 is satisfied """
        if new_accuracy - (30.0 / sample_size) > self.__parameter.accuracy:
            self.__parameter.accuracy = new_accuracy
            self.__logger.info("New maximum accuracy=.2f% reached => Setting parameters:"
                               "\n\t- learning_rate=%.4f"
                               "\n\t- epochs=%d"
                               "\n\t- batch_size=%d",
                               self.parameter.accuracy,
                               self.parameter.learning_rate,
                               self.parameter.epochs,
                               self.parameter.batch_size)
            self.__is_accuracy_improved = True
        else:
            self.__is_accuracy_improved = False

    def __read_file(self):
        with open(self.__file, 'r') as hyper_file:
            return json.load(hyper_file)

    def update_file_if_accuracy_improved(self):
        if self.is_accuracy_improved:
            self.__logger.info("Updating: {}".format(self.__file))
            with open(self.__file, 'w') as hyper_file:
                self.__parameter.accuracy = float(round(self.__parameter.accuracy, 3))
                dict_tmp = self.__parameter.to_dict()
                json.dump(dict_tmp, hyper_file)
        else:
            self.__logger.info("Keep: {} since accuracy did not improve!".format(self.__file))
