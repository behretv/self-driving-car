# **Finding Lane Lines on the Road** 

## Writeup
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. Converting the RGB image to grayscale image
2. Smoothing the image by using Gaussian blur 
3. Detecting edges using the canny operator
4. Filter edges by defining a region of interest
5. Detecting lines using the hought transformation

In order to draw a single line on the left and right lanes, I introduced two new functions:
```pyhton
def seperate_lines(lines, x_threshold):
    """
    Function to sperate the lines into right_lines and left_lines
    """
    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1)) 
            # Slope of 0.5/-0.5 means delta_y shoulde be at least half delta_x, this 
            # threshold helps to prevent considering "too" horizontal lines
            if slope > 0.5 and min([x1, x2]) > x_threshold:
                left_lines.append([x1, y1, x2, y2])
            elif slope < -0.5 and max([x1, x2]) < x_threshold:
                right_lines.append([x1, y1, x2, y2])
```
and 
```pyhton
def extrapolate(lines, img_height):
    """
    Helper function to extrapolate x-value based on the averaged lines and a new y-value
    """
    pos = np.mean(lines, axis=0).astype(int)
    x1 = pos[0]
    y1 = pos[1]
    x2 = pos[2]
    y2 = pos[3]
    y_top = int(img_height/2 * 1.2)
    y_bot = int(img_height)
    x_top = int((y_top - y1) / (y2 - y1) * (x2 - x1) + x1)
    x_bot = int((y_bot - y1) / (y2 - y1) * (x2 - x1) + x1)
    return [[x_top, y_top, x_bot, y_bot]]
```
I am drawing the extra-polated lines within hough_lines and I added the filtering by the slope to draw_lines as well. In this way I could debug the extra polation more easily, since I could draw the extra polated lines and the detected lines in different colors without a bigger effort (see image below).
![img][./test_images_output/solidWhiteCurve.jpg]



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when roards are strongly curved, lines might not be detected and the interpolation might very inaccurate.

Another shortcoming could be the polution of the detected edges with outliers, from illuminations of the road.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to consider the curvature into the extrapolation of the lines, buy dividing it into multiple segments.

Another potential improvement could be to the detection of outliers inside the list of edges, this could be done using RANSAC.
