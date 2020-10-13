## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free 
to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/0.png
[image3]: ./output_images/1.png
[image4]: ./output_images/2.png
[image5]: ./output_images/3.png
[image6]: ./output_images/4.png

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion 
corrected calibration image.

The code for this step is contained in the file `camera_calibration.py` located in 
`./advanced_lane_line/camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy 
of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the 
(x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using 
the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the 
`cv2.undistort()` function and obtained this result: 

![Distortion correction chessboard pattern][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
From the `camera_calibration` function the camera matrix `mtx` and the distortion coefficients `dst` are needed to 
undistort the image 
```python
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

#### 2. Color Transforms and Gradients
 thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 50 through 
58 in `pipeline.py`): 

```python
ksize = 3  # Choose a larger odd number to smooth gradient measurements
gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(np.pi / 3, np.pi / 1.5))
s_binary = hls_select(undist, thresh=(100, 255), channel=2)

combined = np.zeros_like(dir_binary)
combined[(s_binary ==1 ) | ((gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
```

Here's an example of my output for this step. 
![alt text][image3]

#### 3. Perspective transform
transformed image.

The code for my perspective transform includes a function called `warp_coordinates()`, which appears in lines 20 through 
27 in the file `pipline.py` (advanced_lane_lines/pipeline.py). The `warp_coordinates()` function takes as inputs an 
image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination 
points in the following manner:

```python
sz = img.shape
dx0 = 250
dx1 = sz[1] / 2 * 0.9
dy1 = sz[0] / 2 * 1.25

src = np.float32([[dx0, sz[0]], [dx1, dy1], [sz[1] - dx1, dy1], [sz[1] - dx0, sz[0]]])
dst = np.float32([[200, sz[0]], [200, 0], [sz[1] - 200, 0], [sz[1] - 200, sz[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 250, 720      | 200,  720        | 
| 576, 450      | 200,  0      |
| 704, 450      | 1080, 0      |
| 1030, 720     | 1080, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
 and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane-line pixels identification and polynomial fitting
At first window based approach is used to identify the lane pixels (see in `advanced_lane_lines/poly_fit_to_lane.py` 
function `find_lane_pixels()` in lines 106 to 186). 
The center of each window is computed by using the
peak of an histogram generated from the previous window. The second order polynomial is fitted through the centers of 
these window by using these lines of code:
```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
After the first iteration the lane pixels are searched around a margin from the polynomial (see `search_around_poly()
 in 79 to 104) from the previous frame, 
since there should not be too harsh changes from frame to frame. Below is the result for a second iteration, which 
contains and initial identification with the window based method and a final identification using the search around the 
polynomial:
![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 90 through 112 in my code in `pipline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 77 through 83 in my code in `pipeline.py` in the function `map_lane()`.  
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project 
video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  
Where will your pipeline likely fail?  What could you do to make it more robust?

The identification of the lane pixels might identify pixels outside the road, if the curves are very harsh and the 
car is driving on the edge of the lane. A possible solution might be a smaller margin for the polynomial search or an 
exclusion of outliers while applying the window/histogram based approach.

