"""
File to pre-process data to provide tensors
"""
import logging
import enum
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class DataType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class PrepareTensors:
    """

    """

    def __init__(self, train_file, test_file, valid_file):
        self.logger = logging.Logger
        self.file = {
            DataType.TRAIN: train_file,
            DataType.TEST: test_file,
            DataType.VALID: valid_file}
        self._features = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])}
        self._labels = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])}

        for key in DataType:
            self.__import_data(key)

    @property
    def feature(self):
        return self._features

    @property
    def label(self):
        return self._labels

    def process(self):
        for key in DataType:
            self.__pre_process_feature(key)

        encoder = LabelBinarizer()
        encoder.fit(self._labels[DataType.TRAIN])
        for key in DataType:
            self.__pre_process_labels(key, encoder)

        logging.info("Processing finished!")

    def __pre_process_feature(self, key: DataType):
        self.__reshape_image_data(key)
        self.__normalize_grayscale(key)

    def __pre_process_labels(self, key: DataType, encoder):
        encoder = LabelBinarizer()
        encoder.fit(self._labels[key])
        self._labels[key] = encoder.transform(self._labels[key])
        self._labels[key] = self._labels[key].astype(np.float32)

    def __import_data(self, key: DataType):
        file_name = self.file[key]
        with open(file_name, mode='rb') as f:
            data = pickle.load(f)

        self._features[key] = data['features']
        self._labels[key] = data['labels']

    @staticmethod
    def __rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __reshape_image_data(self, key: DataType):
        list_gray = []
        for rgb in self._features[key]:
            list_gray.append(PrepareTensors.__rgb2gray(rgb).flatten())
        self._features[key] = np.array(list_gray)

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
        self._features[key] = a + (((self._features[key] - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))
