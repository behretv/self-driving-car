import csv
import os
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Lambda, Flatten, Dense, Cropping2D,
                          Convolution2D,
                          Dropout)
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

LIST_DATA_FULL = [
    './data/clockwise-1/',
    './data/clockwise-2/',
    './data/clockwise-3/',
    './data/anti-clockwise/',
    './data/curves/',
    './data/recover/',
    './data/data-12-11-20/',
]

DATA_NEW = [
    './data/data/',
]

transfer = True


def load_data(data_input):
    car_images = []
    steering_angles = []
    correction = 0.2  # steering correction for left/right image
    for row, img_file_path in data_input:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = cv2.imread(img_file_path + row[0].split('/')[-1])
        img_left = cv2.imread(img_file_path + row[1].split('/')[-1])
        img_right = cv2.imread(img_file_path + row[2].split('/')[-1])

        # Argument data
        img_center_f = np.fliplr(img_center)
        img_left_f = np.fliplr(img_left)
        img_right_f = np.fliplr(img_right)
        steering_center_f = -1.0 * steering_center
        steering_left_f = -1.0 * steering_left
        steering_right_f = -1.0 * steering_right

        # add images and angles to data set
        car_images.extend([img_center, img_left, img_right])
        car_images.extend([img_center_f, img_left_f, img_right_f])
        steering_angles.extend([steering_center, steering_left, steering_right])
        steering_angles.extend([steering_center_f, steering_left_f, steering_right_f])
    return np.array(car_images), np.array(steering_angles)


def generator(samples, batch_sz=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_sz):
            batch_samples = samples[offset:offset + batch_sz]

            tmp_features, tmp_labels = load_data(batch_samples)
            yield shuffle(tmp_features, tmp_labels)


print("Transfer learning: {}".format(transfer))
if transfer:
    list_data = DATA_NEW
else:
    list_data = LIST_DATA_FULL

print("Loading CSV file...")
data = []
for path in list_data:
    csv_file = path + 'driving_log.csv'
    img_path = path + 'IMG/'
    assert os.path.isfile(csv_file), "No such file {}".format(csv_file)
    assert os.path.isdir(img_path), "No such folder {}".format(img_path)
    with open(csv_file) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data.append((line, img_path))
print("Number of lines: {}".format(len(data)))

print("Loading samples...")
features, labels = load_data(data)
print("Number of samples: {}".format(len(labels)))

train_samples, validation_samples = train_test_split(data, test_size=0.2)

# Set our batch size
batch_size = 128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_sz=batch_size)
validation_generator = generator(validation_samples, batch_sz=batch_size)

features_shape = features.shape[1:]
print("Features shape: {}".format(features_shape))

# Load model for transfer learning or create from scratch
if transfer:
    model = load_model('model.h5')
else:
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=features_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    # Using adam optimizer to avoid necessary learning rate fine tuning
    model.compile(loss='mse', optimizer='adam')

# Train model
print("Training model...")
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=ceil(len(train_samples) / batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=ceil(len(validation_samples) / batch_size),
                                     epochs=5, verbose=1)
print("done!")

print("Saving model...")
model.save('model.h5')
print("done!")

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
