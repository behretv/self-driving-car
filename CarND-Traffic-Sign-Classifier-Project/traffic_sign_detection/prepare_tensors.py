"""
File to pre-process data to provide tensors
"""
import enum
import logging
import pickle

import numpy as np
from sklearn.preprocessing import LabelBinarizer


class DataType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class PrepareTensors:
    """
    Class to process pickled data into tensors
    """

    def __init__(self, train_file, test_file, valid_file):
        self.logger = logging.Logger
        self.file = {
            DataType.TRAIN: train_file,
            DataType.TEST: test_file,
            DataType.VALID: valid_file
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

        self.n_features = 0
        self.n_labels = 0

        for key in DataType:
            self.__import_data(key)

    @property
    def feature(self):
        max_train = np.max(self._feature[DataType.TRAIN])
        min_train = np.min(self._feature[DataType.TRAIN])
        mean_train = np.mean(self._feature[DataType.TRAIN])
        median_train = np.median(self._feature[DataType.TRAIN])
        assert len(self._feature) > 0, "Number of features <= 0!"
        return self._feature

    @property
    def label(self):
        assert len(self._label) > 0, "Number of labels <= 0!"
        return self._label

    def process(self):
        for key in DataType:
            self.__pre_process_feature(key)

        encoder = LabelBinarizer()
        encoder.fit(self._label[DataType.TRAIN])
        for key in DataType:
            self.__pre_process_labels(key, encoder)

        logging.info("Processing finished!")

        self.n_features = self.feature[DataType.TRAIN].shape[1]
        self.n_labels = self.label[DataType.TRAIN].shape[1]

        logging.info("Number of features = %d", self.n_features)
        logging.info("Number of labels = %d", self.n_labels)

    def __pre_process_feature(self, key: DataType):
        #self.__reshape_image_data(key)
        #self.__normalize_grayscale(key)
        pass

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
            list_gray.append(PrepareTensors.__rgb2gray(rgb).flatten())
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
