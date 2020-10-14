import cv2
import numpy as np


class PolyFitToLane:
    def __init__(self):
        self.margin_poly = 80
        self.margin_window = 100  # Set the width of the windows +/- margin
        self.nwindows = 9  # Choose the number of sliding windows
        self.minpix = 50  # Set minimum number of pixels found to recenter window

        self.sane_left = np.array([])
        self.sane_right = np.array([])
        self.ploty = np.array([])
        self.poly_left = []
        self.poly_right = []

        self.img_sz = []
        self.nonzerox = np.array([])
        self.nonzeroy = np.array([])
        self.lane_inds_left = []
        self.lane_inds_right = []

        self.out_img = None

    def get_img(self):
        assert self.out_img is not None
        return self.out_img

    def process(self, warped):
        self.img_sz = warped.shape
        self.ploty = np.linspace(0, self.img_sz[0] - 1, self.img_sz[0])

        # Grab activated pixels
        nonzero = warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        if len(self.poly_left) > 0 and len(self.poly_right) > 0:
            self.search_around_poly()
            self.compute_polynomes()
            self.compute_polynom_coordinates()
            self.image_poly_search(warped)
        else:
            self.find_lane_pixels(warped)
            self.compute_polynomes()
            self.compute_polynom_coordinates()

        return self.sane_left, self.sane_right, self.ploty

    def set_sane_lane(self, left_fitx, right_fitx, threshold=100):
        if len(self.sane_left) == 0:
            self.sane_left = left_fitx
        elif np.average(np.abs(left_fitx - self.sane_left)) < threshold:
            self.sane_left = left_fitx

        if len(self.sane_right) == 0:
            self.sane_right = right_fitx
        elif np.average(np.abs(right_fitx - self.sane_right)) < threshold:
            self.sane_right = right_fitx

    def compute_polynomes(self):
        leftx = self.nonzerox[self.lane_inds_left]
        lefty = self.nonzeroy[self.lane_inds_left]
        rightx = self.nonzerox[self.lane_inds_right]
        righty = self.nonzeroy[self.lane_inds_right]
        self.poly_left = np.polyfit(lefty, leftx, 2)
        self.poly_right = np.polyfit(righty, rightx, 2)

    def compute_polynom_coordinates(self):
        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = self.poly_left[0] * self.ploty ** 2 + self.poly_left[1] * self.ploty + self.poly_left[2]
        right_fitx = self.poly_right[0] * self.ploty ** 2 + self.poly_right[1] * self.ploty + self.poly_right[2]
        self.set_sane_lane(left_fitx, right_fitx)

    def image_detected_lane(self, warped, ploty):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([self.sane_left, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.sane_right, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return color_warp

    def image_poly_search(self, binary_warped):
        # Visualization
        # Create an image to draw on and an image to show the selection window
        img_tmp = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(img_tmp)
        # Color in left and right line pixels
        img_tmp[self.nonzeroy[self.lane_inds_left], self.nonzerox[self.lane_inds_left]] = [255, 0, 0]
        img_tmp[self.nonzeroy[self.lane_inds_right], self.nonzerox[self.lane_inds_right]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.sane_left - self.margin_poly, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.sane_left + self.margin_poly,
                                                                        self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.sane_right - self.margin_poly, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.sane_right + self.margin_poly,
                                                                         self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        self.out_img = cv2.addWeighted(img_tmp, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        pts_left = np.array([self.sane_left, self.ploty], np.int32).T.reshape((-1, 1, 2))
        pts_right = np.array([self.sane_right, self.ploty], np.int32).T.reshape((-1, 1, 2))
        self.out_img = cv2.polylines(self.out_img, [pts_left], color=(255, 255, 0), thickness=2, isClosed=False)
        self.out_img = cv2.polylines(self.out_img, [pts_right], color=(255, 255, 0), thickness=2, isClosed=False)

    def search_around_poly(self):
        # Set the area of search based on activated x-values
        x_left = self.poly_left[0] * (self.nonzeroy ** 2) + \
                 self.poly_left[1] * self.nonzeroy + \
                 self.poly_left[2]
        x_right = self.poly_right[0] * (self.nonzeroy ** 2) + \
                  self.poly_right[1] * self.nonzeroy + \
                  self.poly_right[2]

        #left_roi = (np.min(x_right) > self.nonzerox) & (self.nonzerox < self.img_sz[1]//2 + 100)
        #right_roi = (np.max(x_left) < self.nonzerox) & (self.nonzerox > self.img_sz[1]//2 - 100)
        left_roi = (np.min(x_right) + 100 > self.nonzerox)
        right_roi = (np.max(x_left) - 100 < self.nonzerox)
        self.lane_inds_left = (self.nonzerox > (x_left - self.margin_poly)) & \
                              (self.nonzerox < (x_left + self.margin_poly)) & left_roi
        self.lane_inds_right = (self.nonzerox > (x_right - self.margin_poly)) & \
                               (self.nonzerox < (x_right + self.margin_poly)) & right_roi

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[self.img_sz[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.img_sz[0] // self.nwindows)

        self.lane_inds_left = []
        self.lane_inds_right = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img_sz[0] - (window + 1) * window_height
            win_y_high = self.img_sz[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - self.margin_window
            win_xleft_high = leftx_current + self.margin_window
            win_xright_low = rightx_current - self.margin_window
            win_xright_high = rightx_current + self.margin_window

            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window ###
            idx_y_in_window = (self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high)
            idx_left_in_window = (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)
            idx_right_in_window = (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)
            good_left_inds = (idx_y_in_window & idx_left_in_window).nonzero()[0]
            good_right_inds = (idx_y_in_window & idx_right_in_window).nonzero()[0]

            # Append these indices to the lists
            self.lane_inds_left.append(good_left_inds)
            self.lane_inds_right.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds > self.minpix):
                leftx_current = int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds > self.minpix):
                rightx_current = int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            self.lane_inds_left = np.concatenate(self.lane_inds_left)
            self.lane_inds_right = np.concatenate(self.lane_inds_right)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

