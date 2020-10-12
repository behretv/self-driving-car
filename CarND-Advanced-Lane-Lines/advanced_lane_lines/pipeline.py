import cv2
import numpy as np

from .image_processing import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select, weighted_img
from .poly import fit_poly, find_lane_pixels, measure_curvature_real, search_around_poly


def pipeline(img, mtx, dist, src, dst, prev_left_fit, prev_right_fit, exit=7):
    # 1 Distortion correction
    imshape = img.shape
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if exit == 0:
        return undist

    # 2 Apply thresholds
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(np.pi / 2, np.pi / 1.5))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    s_binary = hls_select(undist, thresh=(170, 255), channel=2)
    combined_binary = np.zeros_like(gradx)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1
    if exit == 1:
        return combined_binary

    # 3 Perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)
    if exit == 2:
        return cv2.warpPerspective(undist, M, combined_binary.shape[::-1], flags=cv2.INTER_LINEAR)

    # 4 Polynomial fit
    if not prev_left_fit or not prev_right_fit:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        left_fitx, right_fitx, ploty = fit_poly(imshape, leftx, lefty, rightx, righty)
    else:
        left_fitx, right_fitx, out_img = search_around_poly(warped, prev_left_fit, prev_right_fit)

    if exit == 3:
        prev_left_fit = np.polyfit(lefty, leftx, 2)
        prev_right_fit = np.polyfit(righty, rightx, 2)
        left_fitx, right_fitx, out_img = search_around_poly(warped, prev_left_fit, prev_right_fit)
        return out_img

    prev_left_fit = left_fitx
    prev_right_fit = right_fitx

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
