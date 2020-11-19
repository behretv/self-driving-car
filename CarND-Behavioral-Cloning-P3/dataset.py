import csv
import cv2
import math
import os
import sklearn

import numpy as np

LIST_DATA_FULL = [
    '/opt/self-driving-car-data/clockwise-1/',
    '/opt/self-driving-car-data/clockwise-2/',
    '/opt/self-driving-car-data/clockwise-3/',
    '/opt/self-driving-car-data/anti-clockwise/',
    '/opt/self-driving-car-data/curves/',
    '/opt/self-driving-car-data/recover/',
    '/opt/self-driving-car-data/data-12-11-20/',
   # '/opt/self-driving-car-data/data/',
]
          
def load_data(data):
	car_images = []
	steering_angles = []
	correction = 0.2 # steering correction for left/right image
	for row, img_path in data:
		steering_center = float(row[3])

		# create adjusted steering measurements for the side camera images
		steering_left = steering_center + correction
		steering_right = steering_center - correction
    
		# read in images from center, left and right cameras
		img_center = cv2.imread(img_path + row[0].split('/')[-1])
		img_left = cv2.imread(img_path + row[1].split('/')[-1])
		img_right = cv2.imread(img_path + row[2].split('/')[-1])
   	 
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


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            X_train, y_train = load_data(batch_samples)
            yield shuffle(X_train, y_train)

list_data = LIST_DATA_FULL
    
print("Loading CSV file...")
data = []
for path in list_data:
	csv_file = path + 'driving_log.csv'
	img_path =  path + 'IMG/'
	assert os.path.isfile(csv_file), "No such file {}".format(csv_file)
	assert os.path.isdir(img_path), "No such folder {}".format(img_folder)
	with open(csv_file) as csv_file:
		reader = csv.reader(csv_file)
		for line in reader:
			data.append((line, img_path))
print("done!")

print("Loading data...")
features, labels = load_data(data)
print("Number of samples: {}".format(len(labels)))
print("done!")
