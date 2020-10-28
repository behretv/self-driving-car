import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def reshape_image_data(image_data):
    list_gray = []
    for rgb in image_data:
        list_gray.append(rgb2gray(rgb).flatten())
    return np.array(list_gray)


def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


def main():
    is_features_normal = False
    is_labels_encod = False

    training_file = "data/train.p"
    validation_file = "data/valid.p"
    testing_file = "data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    train_features, train_labels = train['features'], train['labels']
    valid_features, valid_labels = valid['features'], valid['labels']
    test_features, test_labels = test['features'], test['labels']

    train_features = reshape_image_data(train_features)
    test_features = reshape_image_data(test_features)
    valid_features = reshape_image_data(valid_features)

    if not is_features_normal:
        train_features = normalize_grayscale(train_features)
        test_features = normalize_grayscale(test_features)
        valid_features = normalize_grayscale(valid_features)
        is_features_normal = True
        print("Normalize grayscale done!")

    if not is_labels_encod:
        # Turn labels into numbers and apply One-Hot Encoding
        encoder = LabelBinarizer()
        encoder.fit(train_labels)
        train_labels = encoder.transform(train_labels)
        test_labels = encoder.transform(test_labels)
        valid_labels = encoder.transform(valid_labels)

        # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
        train_labels = train_labels.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
        valid_labels = valid_labels.astype(np.float32)
        is_labels_encod = True
        print("Encoding done!")

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
