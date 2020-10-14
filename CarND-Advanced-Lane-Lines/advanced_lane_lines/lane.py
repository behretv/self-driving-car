import cv2
import numpy as np


class Lane:

    def __init__(self):
        self.img_sz = []
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        self.leftx = np.array([])
        self.rightx = np.array([])
        self.ploty = np.array([])

        self.list_curvatures = []
        self.left_curvature = 5000
        self.right_curvature = 5000
        self.distance = 0
        self.xoffset = 0

    def set_geometry(self, sz, leftx, rightx, ploty):
        self.img_sz = sz
        self.leftx = leftx
        self.rightx = rightx
        self.ploty = ploty

    def measure_curvature_real(self):
        """
        Calculates the curvature of polynomial functions in meters.
        """
        # Define conversions in x and y from pixels space to meters
        left_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.rightx * self.xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty) * self.ym_per_pix
        l_a = left_fit_cr[0]
        l_b = left_fit_cr[1]
        r_a = right_fit_cr[0]
        r_b = right_fit_cr[1]
        self.left_curvature = ((1 + ((2 * l_a * y_eval + l_b) ** 2)) ** 1.5) / np.abs(2 * l_a)
        self.right_curvature = ((1 + ((2 * r_a * y_eval + r_b) ** 2)) ** 1.5) / np.abs(2 * r_a)

        if self.left_curvature > 5000:
            self.left_curvature = 5000
        if self.right_curvature > 5000:
            self.right_curvature = 5000

        # Distance from the center
        self.xoffset = np.mean([self.leftx[-1], self.rightx[-1]]) - self.img_sz[1] / 2
        self.distance = np.abs(self.xoffset) * self.xm_per_pix
        self.list_curvatures.append(np.mean([self.left_curvature, self.right_curvature]))

    def add_text(self, img_color):
        th = 3
        font = cv2.FONT_HERSHEY_PLAIN
        color = (255, 255, 255)
        str_left = 'Curvature left={:.2f}m'.format(self.left_curvature)
        str_right = 'Curvature right={:.2f}m'.format(self.right_curvature)
        str_distance = 'distance={:.2f}m'.format(self.distance)
        cv2.putText(img_color, str_left, org=(50, 50), thickness=th, color=color, fontScale=2, fontFace=font)
        cv2.putText(img_color, str_right, org=(50, 100), thickness=th, color=color, fontScale=2, fontFace=font)
        cv2.putText(img_color, str_distance, org=(50, 150), thickness=th, color=color, fontScale=2, fontFace=font)
