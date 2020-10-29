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

    model_session_file = 'data/parameters/train_model.ckpt'
    training_file = 'data/input/train.p'
    validation_file = 'data/input/test.p'
    testing_file = 'data/input/valid.p'

    logging.basicConfig(level=logging.INFO)

    # 1 Data handling
    data = DataHandler(training_file, testing_file, validation_file)
    #data.process()
    #data.show_random_image()
    feature_valid = data.feature[DataType.VALID]
    label_valid = data.label[DataType.VALID]
    feature_train, label_train = data.get_shuffled_data(DataType.TRAIN)

    # 2 Placeholders
    tf_feature = tf.placeholder(tf.float32, (None, 32, 32, 3))
    tf_label = tf.placeholder(tf.int32, None)

    # 3 DNN
    deep_network = DeepNeuralNetwork(tf_feature)
    deep_network.process()
    logits = deep_network.logits
    optimizer = deep_network.generate_optimizer(tf_label, data, learning_rate)

    # 4 Optimizers
    one_shot_y = tf.one_hot(tf_label, data.n_labels)
    prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_shot_y, 1))
    cost = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # 5 Run and save session
    accuracy = run_session(data, batch_size, epochs, optimizer, tf_feature, tf_label, model_session_file, cost)
    print("Final Accuracy = {:.3f}".format(accuracy))

    # Runs saved session
    saver = tf.train.Saver()
    feature_test, label_test = data.get_shuffled_data(DataType.TEST)
    with tf.Session() as sess:
        saver.restore(sess, model_session_file)
        test_accuracy = sess.run(cost, feed_dict={tf_feature: feature_test, tf_label: label_test})

    print('Test Accuracy: {}'.format(test_accuracy))


def run_session(data, batch_size, epochs, optimizer, tf_feature, tf_label, model_session_file, cost):
    feature_valid = data.feature[DataType.VALID]
    label_valid = data.label[DataType.VALID]
    feature_train, label_train = data.get_shuffled_data(DataType.TRAIN)

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
                sess.run(optimizer, feed_dict={tf_feature: batch_x, tf_label: batch_y})

            accuracy = evaluate(feature_valid, label_valid, tf_feature, tf_label, cost, batch_size)

        saver.save(sess, model_session_file)
        print("Model saved")
    return accuracy


def evaluate(valid_features, valid_labels, tf_features, tf_labels, cost, tmp_batch_size):
    n_features = len(valid_features)
    total_accuracy = 0
    tmp_sess = tf.get_default_session()
    for i_start in range(0, n_features, tmp_batch_size):
        i_end = i_start + tmp_batch_size
        tmp_features = valid_features[i_start:i_end]
        tmp_labels = valid_labels[i_start:i_end]
        tmp_accuracy = tmp_sess.run(cost, feed_dict={tf_features: tmp_features, tf_labels: tmp_labels})
        total_accuracy += (tmp_accuracy * len(tmp_features))
    return total_accuracy / n_features


if __name__ == "__main__":
    main()
