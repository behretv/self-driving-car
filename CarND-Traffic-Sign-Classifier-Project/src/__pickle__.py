"""
Script to convert images to pickle data files
"""

import os
import pickle
import numpy as np
from PIL import Image


PATH_DATA = '../images/internet/test'
PICKLE_FILE = '../data/internet.p'

FILES = os.listdir(PATH_DATA)

FEATURES = np.array([])
LABELS = np.array([])

for file in FILES:
    feature = np.array(Image.open(os.path.join(PATH_DATA, file)))
    label = file[6:8]
    FEATURES = np.append(FEATURES, feature).reshape(-1, 32, 32, 3)
    LABELS = np.append(LABELS, int(label))

DATA = {'features': FEATURES, 'labels': LABELS}

with open(PICKLE_FILE, 'wb') as f:
    pickle.dump(DATA, f)

with open(PICKLE_FILE, 'rb') as f:
    NEW_DATA = pickle.load(f)

assert NEW_DATA is not None
assert 'features' in NEW_DATA
assert 'labels' in NEW_DATA

print("Creating pickle file done!")
