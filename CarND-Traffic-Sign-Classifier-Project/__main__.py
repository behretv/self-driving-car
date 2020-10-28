import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable
from traffic_sign_detection.prepare_tensors import PrepareTensors, DataType


def main():

    training_file = "data/train.p"
    validation_file = "data/valid.p"
    testing_file = "data/test.p"
    tensors = PrepareTensors(training_file, testing_file, validation_file)
    tensors.process()
    train_features = tensors.feature[DataType.TRAIN]
    test_features = tensors.feature[DataType.TRAIN]
    valid_features = tensors.feature[DataType.TRAIN]
    train_labels = tensors.label[DataType.TRAIN]
    test_labels = tensors.label[DataType.TRAIN]
    valid_labels = tensors.label[DataType.TRAIN]

    n_train = len(train_labels)
    n_validation = len(valid_labels)
    n_test = len(test_labels)
    n_features = train_features.shape[1]
    n_labels = train_labels.shape[1]

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_validation)
    print("Number of features =", n_features)
    print("Number of classes =", n_labels)

    # Problem 2 - Set the features and labels tensors
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # Problem 2 - Set the weights and biases tensors
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    biases = tf.Variable(tf.zeros(n_labels))

    assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
    assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases

    prediction = tf.nn.softmax(logits)

    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

    # Training loss
    loss = tf.reduce_mean(cross_entropy)

    # Create an operation that initializes all variables
    init = tf.global_variables_initializer()

    # Test Cases
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_feed_dict)
        session.run(loss, feed_dict=valid_feed_dict)
        session.run(loss, feed_dict=test_feed_dict)
        biases_data = session.run(biases)

    assert not np.count_nonzero(biases_data), 'biases must be zeros'

    print('Tests Passed!')


if __name__ == "__main__":
    main()
