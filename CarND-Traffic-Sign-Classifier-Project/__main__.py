# Load pickled data
import math
import logging
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from traffic_sign_detection.data_handler import DataHandler, DataType
from traffic_sign_detection.deep_neural_network import DeepNeuralNetwork


def main():
    epochs = 10
    batch_size = 128
    learning_rate = 0.001

    training_file = 'data/train.p'
    validation_file = 'data/test.p'
    testing_file = 'data/valid.p'

    logging.basicConfig(level=logging.INFO)
    data = DataHandler(training_file, testing_file, validation_file)
    #data.process()
    #data.show_random_image()
    feature_valid = data.feature[DataType.VALID]
    label_valid = data.label[DataType.VALID]

    feature_train, label_train = data.get_shuffled_data(DataType.TRAIN)

    tf_feature = tf.placeholder(tf.float32, (None, 32, 32, 3))
    deep_network = DeepNeuralNetwork(tf_feature)
    logits = deep_network.process()

    tf_label = tf.placeholder(tf.int32, None)
    one_shot_y = tf.one_hot(tf_label, data.n_labels)

    training_operation = define_optimize_function(one_shot_y, logits, learning_rate)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_shot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_count = int(math.ceil(len(feature_train) / batch_size))

        print("Training...")
        accuracy = 0.0
        for i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count),
                                desc='Previous Accuracy={:.3f} Epoch {:>2}/{}'.format(accuracy, i + 1, epochs),
                                unit='batches')

            feature_train, label_train = shuffle(feature_train, label_train)
            for batch_i in batches_pbar:
                batch_start = batch_i * batch_size
                batch_end = batch_start + batch_size
                batch_x = feature_train[batch_start:batch_end]
                batch_y = label_train[batch_start:batch_end]
                sess.run(training_operation, feed_dict={tf_feature: batch_x, tf_label: batch_y})

            accuracy = evaluate(feature_valid, label_valid, tf_feature, tf_label, accuracy_operation, batch_size)

        print("Final Accuracy = {:.3f}".format(accuracy))
        saver.save(sess, './lenet')
        print("Model saved")


def define_optimize_function(one_shot_y, logits, learning_rate):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_shot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    return training_operation


def evaluate(valid_features, valid_labels, tf_features, tf_labels, accuracy_operation, tmp_batch_size):
    n_features = len(valid_features)
    total_accuracy = 0
    tmp_sess = tf.get_default_session()
    for i_start in range(0, n_features, tmp_batch_size):
        i_end = i_start + tmp_batch_size
        tmp_features = valid_features[i_start:i_end]
        tmp_labels = valid_labels[i_start:i_end]
        tmp_accuracy = tmp_sess.run(accuracy_operation, feed_dict={tf_features: tmp_features, tf_labels: tmp_labels})
        total_accuracy += (tmp_accuracy * len(tmp_features))
    return total_accuracy / n_features


if __name__ == "__main__":
    main()
