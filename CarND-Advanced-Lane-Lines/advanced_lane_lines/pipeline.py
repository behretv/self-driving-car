import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processing import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select, weighted_img
from .poly import fit_poly, find_lane_pixels, measure_curvature_real, search_around_poly
from advanced_lane_lines.poly import PolyFitToLane


class Pipeline:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.src = []
        self.dst = []
        self.prev_left_fit = []
        self.prev_right_fit = []
        self.poly = PolyFitToLane()

    def set_warp_coordinates(self, img):
        sz = img.shape
        dx1 = sz[1] / 2 * 0.9
        dy1 = sz[0] / 2 * 1.25
        dx0 = 250

        self.src = np.float32([[dx0, sz[0]], [dx1, dy1], [sz[1] - dx1, dy1], [sz[1] - dx0, sz[0]]])
        self.dst = np.float32([[200, sz[0]], [200, 0], [sz[1] - 200, 0], [sz[1] - 200, sz[0]]])

    def process_image(self, img, output_name, exit_loop):
        out = self.pipeline(img, exit_loop=exit_loop)

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Input')
        ax2.set_title('Output')
        ax1.imshow(img)
        x = self.src[:, 0]
        y = self.src[:, 1]
        ax1.plot(x, y, 'b--', lw=2)
        ax2.imshow(out)
        plt.savefig(output_name)

    def pipeline(self, img, exit_loop=7):
        # 1 Distortion correction
        imshape = img.shape
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if exit_loop == 0:
            return undist

        # 2 Apply thresholds
        ksize = 3  # Choose a larger odd number to smooth gradient measurements
        gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(np.pi / 3, np.pi / 1.5))
        combined = np.zeros_like(dir_binary)
        combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        s_binary = hls_select(undist, thresh=(100, 255), channel=2)
        combined_binary = np.zeros_like(gradx)
        combined_binary[(s_binary == 1) | (combined == 1)] = 1
        if exit_loop == 1:
            return combined_binary

        # 3 Perspective transformation
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)
        if exit_loop == 2:
            return cv2.warpPerspective(undist, M, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)

        # 4 Polynomial fit
        left_fitx, right_fitx, ploty, out_img = self.poly.process(warped)
        if exit_loop == 3:
            return []

        # 5 Calculate radius
        left_cur, right_cur = measure_curvature_real(ploty, left_fitx, right_fitx)
        print('l={} \t r={}'.format(left_cur, right_cur))

        # 6 Transform back
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        unwarped = cv2.warpPerspective(color_warp, Minv, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)

        return weighted_img(unwarped, img)
