import math
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.ops.variables import Variable
from traffic_sign_detection.prepare_tensors import PrepareTensors, DataType


def main():
    logging.basicConfig(level=logging.INFO)

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

    # Problem 2 - Set the features and labels tensors
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    logits = tf.matmul(features, tensors.weights) + tensors.bias

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
        biases_data = session.run(tensors.bias)

    assert not np.count_nonzero(biases_data), 'biases must be zeros'

    print('Tests Passed!')

    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    print('Accuracy function created.')

    configuration = 2
    epochs = 3
    batch_size = 500
    learning_rate = 0.2
    # Configuration 1
    if configuration == 1:
        epochs = 3
        batch_size = 1000
        learning_rate = 0.05
    elif configuration == 2:
        epochs = 5
        batch_size = 800
        learning_rate = 0.1
    elif configuration == 3:
        epochs = 5
        batch_size = 800
        learning_rate = 0.1

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(validation_accuracy)

            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

    print('Validation accuracy at {}'.format(validation_accuracy))


if __name__ == "__main__":
    main()
