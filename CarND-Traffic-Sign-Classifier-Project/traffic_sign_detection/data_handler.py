"""
File to pre-process parameter to provide tensors
"""
import enum
import pickle
import random
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn import utils
from traffic_sign_detection.file_handler import FileHandler


class DataType(enum.Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class DataHandler:
    """
    Class to process pickled parameter into tensors
    """

    def __init__(self, files: FileHandler):
        self.__logger = logging.getLogger(DataHandler.__name__)
        self.__file = {
            DataType.TRAIN: files.training_file,
            DataType.TEST: files.testing_file,
            DataType.VALID: files.validation_file
        }
        self.__feature = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])
        }
        self.__label = {
            DataType.TRAIN: np.array([]),
            DataType.TEST: np.array([]),
            DataType.VALID: np.array([])
        }

        for key in DataType:
            self.__import_data(key)

        self.__image_shape = self.feature[DataType.TRAIN].shape[1:]
        self.__number_of_labels = len(set(self.label[DataType.TRAIN]))
        self.__logger.info("Number of features = %d", self.image_shape[1] * self.image_shape[2])
        self.__logger.info("Number of labels = %d", self.n_labels)

    @property
    def feature(self):
        assert len(self.__feature) > 0, "Number of features <= 0!"
        return self.__feature

    @property
    def label(self):
        assert len(self.__label) > 0, "Number of labels <= 0!"
        return self.__label

    @property
    def n_labels(self):
        return self.__number_of_labels

    @property
    def image_depth(self):
        depth = 1
        if len(self.__image_shape) == 3:
            depth = self.__image_shape[2]
        return depth

    @property
    def image_shape(self):
        return (None,) + self.__image_shape

    def sample_size(self, key):
        return len(self.feature[key])

    def get_shuffled_data(self, key: DataType):
        feature_train = self.feature[key]
        label_train = self.label[key]
        return utils.shuffle(feature_train, label_train)

    def process(self):
        for key in DataType:
            self.__pre_process_feature(key)

       # encoder = LabelBinarizer()
       # encoder.fit(self.__label[DataType.TRAIN])
       # for key in DataType:
       #     self.__pre_process_labels(key, encoder)

        self.__image_shape = self.feature[DataType.TRAIN].shape[1:]
        self.__logger.info("Processing finished!")

    def visualize_random_image(self):
        feature_train = self.feature[DataType.TRAIN]
        label_train = self.label[DataType.TRAIN]
        index = random.randint(0, len(feature_train))
        image = feature_train[index].squeeze()
        gray = self.__rgb2gray(image)

        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Example RGB image for Label " + str(label_train[index]))
        plt.subplot(122)
        plt.imshow(gray, cmap='gray')
        plt.title("Example gray-scale image for Label " + str(label_train[index]))
        plt.show()

    def visualize_labels_histogram(self):
        labels_train = self.label[DataType.TRAIN]
        labels_test = self.label[DataType.TEST]
        labels_valid = self.label[DataType.VALID]

        bins = range(0, self.n_labels)

        plt.hist([labels_train, labels_test, labels_valid], bins, label=['training', 'test', 'validation'])
        plt.title("Labels Histogram")
        plt.xlim(bins[0], bins[-1])
        plt.legend(loc='upper center')
        plt.show()

    def __pre_process_feature(self, key: DataType):
        self.__reshape_image_data(key)
        self.__normalize_grayscale(key)

    def __pre_process_labels(self, key: DataType, encoder):
        self.__label[key] = encoder.transform(self.__label[key])
        self.__label[key] = self.__label[key].astype(np.float32)

    def __import_data(self, key: DataType):
        file_name = self.__file[key]
        with open(file_name, mode='rb') as f:
            data = pickle.load(f)

        self.__feature[key] = data['features']
        self.__label[key] = data['labels']

        self.__logger.info("Number of {} examples ={}".format(key.value, self.sample_size(key)))

    @staticmethod
    def __rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __reshape_image_data(self, key: DataType):
        list_gray = []
        for rgb in self.__feature[key]:
            list_gray.append(DataHandler.__rgb2gray(rgb))
        tmp_np_array = np.array(list_gray)
        self.__feature[key] = tmp_np_array.reshape(tmp_np_array.shape + (1,))

    def __normalize_grayscale(self, key: DataType):
        """
        Normalize the image parameter with Min-Max scaling to a range of [0.1, 0.9]
        :param key: 'test', 'train' or 'valid'
        :return: Normalized image parameter
        """
        a = 0.1
        b = 0.9
        grayscale_min = 0
        grayscale_max = 255
        self.__feature[key] = a + (((self.__feature[key] - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))
