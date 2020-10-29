# Load pickled data
import math
import pickle
import matplotlib.pyplot as plt
import random
import logging
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from tqdm import tqdm

from traffic_sign_detection.prepare_tensors import PrepareTensors, DataType

training_file = 'data/train.p'
validation_file = 'data/test.p'
testing_file = 'data/valid.p'

logging.basicConfig(level=logging.INFO)
tensors = PrepareTensors(training_file, testing_file, validation_file)
# tensors.process()
X_train = tensors.feature[DataType.TRAIN]
y_train = tensors.label[DataType.TRAIN]
X_test = tensors.feature[DataType.TEST]
y_test = tensors.label[DataType.TEST]
X_valid = tensors.feature[DataType.VALID]
y_valid = tensors.label[DataType.VALID]

# Total number of images: 51839
n_train = 34799

n_validation = 12630

n_test = 4410

image_shape = [32, 32]

n_classes = 43

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.imshow(image)
plt.show()
print(y_train[index])

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 10
BATCH_SIZE = 128
FEATURE = tf.placeholder(tf.float32, (None, 32, 32, 3))
LABEL = tf.placeholder(tf.int32, None)


def le_net_5(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    # B_1 = tf.Variable(tf.truncated_normal(shape=(1, 28, 28, 6), mean=mu, stddev=sigma))
    B_1 = tf.Variable(tf.zeros(shape=(1, 28, 28, 6)))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer_1 = tf.nn.conv2d(x, W_1, strides=strides, padding=padding) + B_1

    # Activation.
    layer_1 = tf.nn.relu(layer_1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_1 = tf.nn.max_pool(layer_1, k, strides, padding)

    # Layer 2: Convolutional. Output = 10x10x16.
    W_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    # B_2 = tf.Variable(tf.truncated_normal(shape=(1, 10, 10, 16), mean=mu, stddev=sigma))
    B_2 = tf.Variable(tf.zeros(shape=(1, 10, 10, 16)))
    strides = [1, 1, 1, 1]
    padding = 'VALID'
    layer_2 = tf.nn.conv2d(layer_1, W_2, strides=strides, padding=padding) + B_2

    # Activation.
    layer_2 = tf.nn.relu(layer_2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

    # Flatten. Input = 5x5x16. Output = 400.
    fc = flatten(layer_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    W_3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    # B_3 = tf.Variable(tf.truncated_normal(shape=(1, 120), mean=mu, stddev=sigma))
    B_3 = tf.Variable(tf.zeros(shape=(1, 120)))
    layer_3 = tf.matmul(fc, W_3) + B_3

    # Activation.
    layer_3 = tf.nn.relu(layer_3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    W_4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    # B_4 = tf.Variable(tf.truncated_normal(shape=(1, 84), mean=mu, stddev=sigma))
    B_4 = tf.Variable(tf.zeros(shape=(1, 84)))
    layer_4 = tf.matmul(layer_3, W_4) + B_4

    # Activation.
    layer_4 = tf.nn.relu(layer_4)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    W_5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    # B_5 = tf.Variable(tf.truncated_normal(shape=(1, 43), mean=mu, stddev=sigma))
    B_5 = tf.Variable(tf.zeros(shape=(1, 43)))
    logits = tf.matmul(layer_4, W_5) + B_5

    return logits


one_shot_y = tf.one_hot(LABEL, 43)

learning_rate = 0.001

logits = le_net_5(FEATURE)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_shot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_shot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(valid_features, valid_labels, x, y):
    n_features = len(valid_features)
    total_accuracy = 0
    tmp_sess = tf.get_default_session()
    for i_start in range(0, n_features, BATCH_SIZE):
        i_end = i_start + BATCH_SIZE
        tmp_features = valid_features[i_start:i_end]
        tmp_labels = valid_labels[i_start:i_end]
        tmp_accuracy = tmp_sess.run(accuracy_operation, feed_dict={x: tmp_features, y: tmp_labels})
        total_accuracy += (tmp_accuracy * len(tmp_features))
    return total_accuracy / n_features


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = X_train.shape[0]

    batch_count = int(math.ceil(len(X_train) / BATCH_SIZE))

    print("Training...")
    accuracy = 0.0
    for i in range(EPOCHS):

        # Progress bar
        batches_pbar = tqdm(range(batch_count),
                            desc='Previous Accuracy={:.3f} Epoch {:>2}/{}'.format(accuracy, i + 1, EPOCHS),
                            unit='batches')

        X_train, y_train = shuffle(X_train, y_train)
        for batch_i in batches_pbar:
            batch_start = batch_i * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE
            batch_x = X_train[batch_start:batch_end]
            batch_y = y_train[batch_start:batch_end]
            sess.run(training_operation, feed_dict={FEATURE: batch_x, LABEL: batch_y})

        accuracy = evaluate(X_valid, y_valid, FEATURE, LABEL)

    print("Final Accuracy = {:.3f}".format(accuracy))
    saver.save(sess, './lenet')
    print("Model saved")
