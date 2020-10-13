import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processing import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select, weighted_img
from advanced_lane_lines.poly_fit_to_lane import PolyFitToLane


class Pipeline:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.src = []
        self.dst = []
        self.prev_left_fit = []
        self.prev_right_fit = []
        self.poly = PolyFitToLane()
        self.list_radius = []
        self.img_sz = []

    def warp_coordinates(self, img):
        sz = img.shape
        dx0 = 200
        dx1 = sz[1] / 2 * 0.9
        dy1 = sz[0] / 2 * 1.25

        self.src = np.float32([[dx0, sz[0]], [dx1, dy1], [sz[1] - dx1, dy1], [sz[1] - dx0, sz[0]]])
        self.dst = np.float32([[200, sz[0]], [200, 0], [sz[1] - 200, 0], [sz[1] - 200, sz[0]]])
        self.img_sz = sz

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
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        if exit_loop == 0:
            return undist

        # 2 Apply thresholds
        ksize = 3  # Choose a larger odd number to smooth gradient measurements
        gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 100))
        dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(np.pi / 3, np.pi / 1.5))
        s_binary = hls_select(undist, thresh=(100, 255), channel=2)

        combined = np.zeros_like(dir_binary)
        combined[(s_binary == 1) | ((gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
        if exit_loop == 1:
            return combined

        # 3 Perspective transformation
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        warped = cv2.warpPerspective(combined, M, combined.shape[::-1], flags=cv2.INTER_LINEAR)
        if exit_loop == 2:
            return cv2.warpPerspective(undist, M, combined.shape[::-1], flags=cv2.INTER_LINEAR)

        # 4 Polynomial fit
        left_fitx, right_fitx, ploty, out_img = self.poly.process(warped)
        if exit_loop == 3:
            return out_img

        # 5 Calculate radius
        curvature, distance = self.measure_curvature_real(ploty, left_fitx, right_fitx)
        self.list_radius.append(curvature)
        if exit_loop == 4:
            cv2.putText(undist, 'curvature={}'.format(curvature),
                        org=(50, 50), thickness=2, color=(255, 255, 255), fontScale=4,
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(undist, 'distance={}'.format(distance),
                        org=(50, 100), thickness=2, color=(255, 255, 255), fontScale=4,
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            return undist

        # 6 Transform back
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        unwarped = cv2.warpPerspective(color_warp, Minv, combined.shape[::-1], flags=cv2.INTER_LINEAR)

        return weighted_img(unwarped, undist)

    def measure_curvature_real(self, ploty, leftx, rightx):
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

        # Distance from the center
        distance = np.abs(np.mean([leftx[-1], rightx[-1]]) - self.img_sz[1] / 2) * xm_per_pix

        return np.mean([left_curverad, right_curverad]), distance
