import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from advanced_lane_lines import pipeline, camera_calibration


def process_image(img_name, exit=7, output_name='img.png'):
    img = mpimg.imread(img_name)
    imshape = img.shape
    dx_top = imshape[1] / 2 * 0.9
    dy_top = imshape[0] / 2 * 1.25
    dx_bot = 150
    dy_bot = imshape[0]
    src = np.float32([[dx_bot, dy_bot],
                      [dx_top, dy_top],
                      [imshape[1] - dx_top, dy_top],
                      [imshape[1] - dx_bot, dy_bot]])
    dst = np.float32([[300, imshape[0]],
                      [300, 0],
                      [imshape[1] - 300, 0],
                      [imshape[1] - 300, imshape[0]]])

    prev_left_fitx = []
    prev_right_fitx = []
    out = pipeline(img, mtx, dist, src, dst, prev_left_fitx, prev_right_fitx, exit=exit)

    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Input')
    ax2.set_title('Output')
    ax1.imshow(img)
    x = src[:, 0]
    y = src[:, 1]
    ax1.plot(x, y, 'b--', lw=2)
    ax2.imshow(out)
    plt.savefig(output_name)


# %matplotlib qt
images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

test_images = glob.glob('./test_images/*.jpg')
process_image(test_images[3], 3, './output_images/3.png')

if False:
    video_name = './project_video.mp4'
    video_clip = VideoFileClip(video_name)
    video_result = video_clip.fl_image(lambda xx: pipeline(xx, mtx, dist, src, dst, [], [], 7))
    video_result.write_videofile('./result.mp4')
    # for frame in video_clip.iter_frames():
    #    result = pipeline(frame, mtx, dist, src, dst)
    #    plt.imshow(result)
    #    plt.pause(0.0001)
