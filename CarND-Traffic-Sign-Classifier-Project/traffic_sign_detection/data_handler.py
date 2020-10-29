"""
File to pre-process data to provide tensors
"""
import enum
import pickle
import random
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn import utils
from traffic_sign_detection.file_handler import FileHandler


class DataType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class DataHandler:
    """
    Class to process pickled data into tensors
    """

    def __init__(self, files: FileHandler):
        self.logger = logging.Logger
        self.file = {
            DataType.TRAIN: files.training_file,
            DataType.TEST: files.testing_file,
            DataType.VALID: files.validation_file
        }
        self._feature = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])
        }
        self._label = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])
        }

        for key in DataType:
            self.__import_data(key)

        self._n_features = self.feature[DataType.TRAIN].shape[1:3]
        self._n_labels = len(set(self.label[DataType.TRAIN]))
        logging.info("Number of features = %d", self.n_features[0]*self.n_features[1])
        logging.info("Number of labels = %d", self.n_labels)

    @property
    def feature(self):
        assert len(self._feature) > 0, "Number of features <= 0!"
        return self._feature

    @property
    def label(self):
        assert len(self._label) > 0, "Number of labels <= 0!"
        return self._label

    @property
    def n_labels(self):
        return self._n_labels

    @property
    def n_features(self):
        return self._n_features

    def get_shuffled_data(self, key: DataType):
        feature_train = self.feature[key]
        label_train = self.label[key]
        return utils.shuffle(feature_train, label_train)

    def process(self):
        for key in DataType:
            self.__pre_process_feature(key)

        encoder = LabelBinarizer()
        encoder.fit(self._label[DataType.TRAIN])
        for key in DataType:
            self.__pre_process_labels(key, encoder)

        logging.info("Processing finished!")

    def show_random_image(self):
        feature_train = self.feature[DataType.TRAIN]
        label_train = self.label[DataType.TRAIN]
        index = random.randint(0, len(feature_train))
        image = feature_train[index].squeeze()

        plt.imshow(image)
        plt.title("Label " + str(label_train[index]))
        plt.show()

    def __pre_process_feature(self, key: DataType):
        self.__reshape_image_data(key)
        self.__normalize_grayscale(key)

    def __pre_process_labels(self, key: DataType, encoder):
        self._label[key] = encoder.transform(self._label[key])
        self._label[key] = self._label[key].astype(np.float32)

    def __import_data(self, key: DataType):
        file_name = self.file[key]
        with open(file_name, mode='rb') as f:
            data = pickle.load(f)

        self._feature[key] = data['features']
        self._label[key] = data['labels']

        n_labels = len(self._label[key])
        logging.info("Number of {} examples ={}".format(key.value, n_labels))

    @staticmethod
    def __rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __reshape_image_data(self, key: DataType):
        list_gray = []
        for rgb in self._feature[key]:
            list_gray.append(DataHandler.__rgb2gray(rgb).flatten())
        self._feature[key] = np.array(list_gray)

    def __normalize_grayscale(self, key: DataType):
        """
        Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
        :param key: 'test', 'train' or 'valid'
        :return: Normalized image data
        """
        a = 0.1
        b = 0.9
        grayscale_min = 0
        grayscale_max = 255
        self._feature[key] = a + (((self._feature[key] - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))
