"""
File to pre-process data to provide tensors
"""
import logging
import enum
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable


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
        self._feed_dict = {
            DataType.TRAIN: {},
            DataType.TEST: {},
            DataType.VALID: {}
        }

        self._bias = None
        self._weights = None
        self._loss = None

        for key in DataType:
            self.__import_data(key)

        logging.info("Number of features = %d", len(self.feature[DataType.TRAIN]))
        logging.info("Number of labels = %d", len(self.label[DataType.TRAIN]))

    @property
    def feature(self):
        return self._feature

    @property
    def label(self):
        return self._label

    @property
    def bias(self):
        assert isinstance(self._bias, Variable), 'biases must be a TensorFlow variable'
        return self._bias

    @property
    def weights(self):
        assert isinstance(self._weights, Variable), 'weights must be a TensorFlow variable'
        return self._weights

    @property
    def loss(self):
        return self._loss

    @property
    def feed_dict(self):
        return self._feed_dict

    @feed_dict.setter
    def feed_dict(self, value):
        self._feed_dict = value

    def process(self):
        for key in DataType:
            self.__pre_process_feature(key)

        encoder = LabelBinarizer()
        encoder.fit(self._label[DataType.TRAIN])
        for key in DataType:
            self.__pre_process_labels(key, encoder)

        logging.info("Processing finished!")

        train_features = self.feature[DataType.TRAIN]
        train_labels = self.label[DataType.TRAIN]

        n_features = train_features.shape[1]
        n_labels = train_labels.shape[1]

        print("Number of features =", n_features)
        print("Number of classes =", n_labels)

        # Problem 2 - Set the features and labels tensors
        features = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32)

        # Problem 2 - Set the weights and biases tensors
        self._weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
        self._bias = tf.Variable(tf.zeros(n_labels))

        # Feed dicts for training, validation, and test session
        for key in DataType:
            self._feed_dict[key] = {features: self.feature[key], labels: self.label[key]}

        # Linear Function WX + b
        logits = tf.matmul(features, self.weights) + self.bias
        prediction = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)
        self._loss = tf.reduce_mean(cross_entropy)

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
