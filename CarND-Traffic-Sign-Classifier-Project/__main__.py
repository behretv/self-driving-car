import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.variables import Variable


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

    n_train = len(train['labels'])
    n_validation = len(valid['labels'])
    n_test = len(test['labels'])
    image_shape = train['features'].shape[1:3]
    n_classes = len(set(train['labels']))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_validation)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    if not is_features_normal:
        train_features = normalize_grayscale(train_features)
        test_features = normalize_grayscale(test_features)
        is_features_normal = True

    if not is_labels_encod:
        # Turn labels into numbers and apply One-Hot Encoding
        encoder = LabelBinarizer()
        encoder.fit(train_labels)
        train_labels = encoder.transform(train_labels)
        test_labels = encoder.transform(test_labels)

        # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
        train_labels = train_labels.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
        is_labels_encod = True

    features_count = 784
    labels_count = 10

    # Problem 2 - Set the features and labels tensors
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # Problem 2 - Set the weights and biases tensors
    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))

    assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
    assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
    assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
    assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

    assert features._shape == None or ( \
                features._shape.dims[0].value is None and \
                features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
    assert labels._shape == None or ( \
                labels._shape.dims[0].value is None and \
                labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
    assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
    assert biases._variable._shape == (10), 'The shape of biases is incorrect'

    assert features._dtype == tf.float32, 'features must be type float32'
    assert labels._dtype == tf.float32, 'labels must be type float32'

    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases

    prediction = tf.nn.softmax(logits)

    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

    # some students have encountered challenges using this function, and have resolved issues
    # using https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    # please see this thread for more detail https://discussions.udacity.com/t/accuracy-0-10-in-the-intro-to-tensorflow-lab/272469/9

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
