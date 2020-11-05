import os
import pickle
import numpy as np
from PIL import Image


PATH_DATA = 'images/internet/test'
PICKLE_FILE = 'data/internet.p'

files = os.listdir(PATH_DATA)

features = np.array([])
labels = np.array([])
for file in files:
    feature = np.array(Image.open(os.path.join(PATH_DATA, file)))
    label = file[6:8]
    features = np.append(features, feature).reshape(-1, 32, 32, 3)
    labels = np.append(labels, int(label))

data = {'features': features, 'labels': labels}

with open(PICKLE_FILE, 'wb') as f:
    pickle.dump(data, f)

with open(PICKLE_FILE, 'rb') as f:
    new_data = pickle.load(f)

assert new_data is not None
assert 'features' in new_data
assert 'labels' in new_data

print("Creating pickle file done!")


