import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processing import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select, weighted_img
from advanced_lane_lines.poly_fit_to_lane import PolyFitToLane
from advanced_lane_lines.lane import Lane


class Pipeline:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.src = []
        self.dst = []
        self.prev_left_fit = []
        self.prev_right_fit = []
        self.poly = PolyFitToLane()
        self.lane = Lane()
        self.img_sz = []

        self.img_color = None
        self.img_binary = None

    def warp_coordinates(self, img):
        sz = img.shape
        dx0 = 200
        dx1 = sz[1] / 2 * 0.9
        dy1 = sz[0] / 2 * 1.25

        self.src = np.float32([[dx0, sz[0]], [dx1, dy1], [sz[1] - dx1, dy1], [sz[1] - dx0, sz[0]]])
        self.dst = np.float32([[200, sz[0]], [200, 0], [sz[1] - 200, 0], [sz[1] - 200, sz[0]]])
        self.img_sz = sz
        self.src[:, 0] += self.lane.xoffset / 2

    def process_image(self, img, output_name, exit_loop):
        out = self.pipeline(img, exit_loop=exit_loop)
        if len(out.shape) == 2:
            color_map = 'gray'
        else:
            color_map = None

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Input')
        ax2.set_title('Output')
        ax1.imshow(img)
        x = self.src[:, 0]
        y = self.src[:, 1]
        ax1.plot(x, y, 'b--', lw=2)
        ax2.imshow(out, cmap=color_map)
        plt.savefig(output_name)

    def pipeline(self, img, exit_loop=7):
        # 1 Distortion correction
        self.img_color = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if exit_loop == 0:
            return self.img_color

        # 2 Apply thresholds
        ksize = 3  # Choose a larger odd number to smooth gradient measurements
        gradx = abs_sobel_thresh(self.img_color, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = mag_thresh(self.img_color, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = dir_threshold(self.img_color, sobel_kernel=ksize, thresh=(np.pi / 3, np.pi / 1.5))
        s_binary = hls_select(self.img_color, thresh=(100, 255), channel=2)

        combined = np.zeros_like(dir_binary)
        combined[(s_binary == 1) | ((gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
        if exit_loop == 1:
            return combined

        # 3 Perspective transformation
        m = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(combined, m, combined.shape[::-1], flags=cv2.INTER_LINEAR)
        if exit_loop == 2:
            return cv2.warpPerspective(self.img_color, m, combined.shape[::-1], flags=cv2.INTER_LINEAR)

        # 4 Polynomial fit
        left_fitx, right_fitx, ploty = self.poly.__pre_process_feature(warped)
        if exit_loop == 3:
            return self.poly.get_img()

        # 5 Calculate radius
        self.lane.set_geometry(self.img_sz, left_fitx, right_fitx, ploty)
        self.lane.measure_curvature_real()
        self.lane.add_text(self.img_color)
        if exit_loop == 4:
            return self.img_color

        # 6 Transform back
        color_warp = self.poly.image_detected_lane(warped, ploty)
        minv = cv2.getPerspectiveTransform(self.dst, self.src)
        unwarped = cv2.warpPerspective(color_warp, minv, combined.shape[::-1], flags=cv2.INTER_LINEAR)

        return weighted_img(unwarped, self.img_color)

