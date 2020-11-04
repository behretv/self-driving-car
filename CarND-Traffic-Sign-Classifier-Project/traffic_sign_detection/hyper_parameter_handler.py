""" Class to collect hyper parameter and optimize them """
import json
import logging

from traffic_sign_detection.file_handler import FileHandler
from traffic_sign_detection.parameters import Parameters


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
        self.__parameter.print()

    def update_accuracy(self, new_accuracy, sample_size):
        """ Update only if rule of 30 is satisfied """
        if new_accuracy - (30.0 / sample_size) > self.__parameter.accuracy:
            self.__parameter.accuracy = new_accuracy
            self.__logger.info("New maximum accuracy=.2f% reached => Setting parameters:", new_accuracy)
            self.__parameter.print()
            self.__is_accuracy_improved = True
        else:
            self.__is_accuracy_improved = False

    def __read_file(self):
        with open(self.__file, 'r') as hyper_file:
            return json.load(hyper_file)

    def update_file(self):
        self.__logger.info("Updating: {}".format(self.__file))
        with open(self.__file, 'w') as hyper_file:
            self.__parameter.accuracy = float(round(self.__parameter.accuracy, 3))
            dict_tmp = self.__parameter.to_dict()
            json.dump(dict_tmp, hyper_file)
