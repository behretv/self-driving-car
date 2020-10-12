import cv2
import numpy as np
import matplotlib.pyplot as plt


class PolyFitToLane:
    def __init__(self):
        self.margin_poly = 100
        self.margin_window = 100  # Set the width of the windows +/- margin
        self.img_sz = []
        self.leftx = []
        self.lefty = []
        self.rightx = []
        self.righty = []
        self.nonzerox = []
        self.nonzeroy = []
        self.left_lane_inds = 0
        self.right_lane_inds = 0
        self.prev_left_fit = []
        self.prev_right_fit = []

    def process(self, warped):
        imshape = warped.shape
        if len(self.prev_left_fit) > 0 and len(self.prev_right_fit) > 0:
            self.leftx, lefty, rightx, righty = self.search_around_poly(warped)
            left_fitx, right_fitx, ploty = self.fit_poly(imshape, lefty, rightx, righty)
            out_img = self.visualize(warped, left_fitx, right_fitx, ploty)
        else:
            self.leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(warped)
            left_fitx, right_fitx, ploty = self.fit_poly(imshape, lefty, rightx, righty)

        return left_fitx, right_fitx, ploty, out_img

    def fit_poly(self, img_shape, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit()
        left_fit = np.polyfit(lefty, self.leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit

        return left_fitx, right_fitx, ploty

    def visualize(self, binary_warped, left_fitx, right_fitx, ploty):
        # Visualization
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin_poly, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin_poly,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin_poly, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin_poly,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        pts_left = np.array([left_fitx, ploty], np.int32).T.reshape((-1, 1, 2))
        pts_right = np.array([right_fitx, ploty], np.int32).T.reshape((-1, 1, 2))
        result = cv2.polylines(result, [pts_left], color=(255, 255, 0), thickness=2, isClosed=False)
        result = cv2.polylines(result, [pts_right], color=(255, 255, 0), thickness=2, isClosed=False)
        return result

    def search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values
        x_left = self.prev_left_fit[0] * (self.nonzeroy ** 2) + \
                 self.prev_left_fit[1] * self.nonzeroy + \
                 self.prev_left_fit[2]
        x_right = self.prev_right_fit[0] * (self.nonzeroy ** 2) + \
                  self.prev_right_fit[1] * self.nonzeroy + \
                  self.prev_right_fit[2]
        left_lane_inds = (self.nonzerox > (x_left - self.margin_poly)) & (self.nonzerox < (x_left + self.margin_poly))
        right_lane_inds = (
                (self.nonzerox > (x_right - self.margin_poly)) & (self.nonzerox < (x_right + self.margin_poly)))

        # Again, extract left and right line pixel positions
        leftx = self.nonzerox[left_lane_inds]
        lefty = self.nonzeroy[left_lane_inds]
        rightx = self.nonzerox[right_lane_inds]
        righty = self.nonzeroy[right_lane_inds]
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        return leftx, lefty, rightx, righty

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - self.margin_window
            win_xleft_high = leftx_current + self.margin_window
            win_xright_low = rightx_current - self.margin_window
            win_xright_high = rightx_current + self.margin_window

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window ###
            idx_y_in_window = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
            idx_left_in_window = (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            idx_right_in_window = (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            good_left_inds = (idx_y_in_window & idx_left_in_window).nonzero()[0]
            good_right_inds = (idx_y_in_window & idx_right_in_window).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds > minpix):
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds > minpix):
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')

        return leftx, lefty, rightx, righty, out_img


def measure_curvature_real(ploty, leftx, rightx):
    """
    Calculates the curvature of polynomial functions in meters.
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) * ym_per_pix
    lA = left_fit_cr[0]
    lB = left_fit_cr[1]
    lC = left_fit_cr[2]
    rA = right_fit_cr[0]
    rB = right_fit_cr[1]
    rC = right_fit_cr[2]
    left_curverad = ((1 + ((2 * lA * y_eval + lB) ** 2)) ** 1.5) / np.abs(2 * lA)
    right_curverad = ((1 + ((2 * rA * y_eval + rB) ** 2)) ** 1.5) / np.abs(2 * rA)

    return left_curverad, right_curverad
