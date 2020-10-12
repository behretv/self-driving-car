import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# %matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(fname)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        plt.imshow(img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

test_images = glob.glob('../test_images/*.jpg')
img = mpimg.imread(test_images[6])
imshape = img.shape
dx_top = imshape[1]/2 * 0.9
dy_top = imshape[0]/2 * 1.25
dx_bot = 150
dy_bot = imshape[0]
src = np.float32([[dx_bot, dy_bot],
                [dx_top, dy_top],
                [imshape[1]-dx_top, dy_top],
                [imshape[1]-dx_bot, dy_bot]])

out = pipeline(img, mtx, dist, src)

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Input image with ROI')

ax1.imshow(img)
x = src[:,0]
y = src[:,1]
ax1.plot(x, y, 'b--', lw=2)

ax2.set_title('Image with detected lane')
ax2.imshow(out)