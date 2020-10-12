import matplotlib.image as mpimg
import glob

from moviepy.video.io.VideoFileClip import VideoFileClip

from advanced_lane_lines import camera_calibration
from advanced_lane_lines.pipeline import Pipeline


# %matplotlib qt
images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

test_images = glob.glob('./test_images/*.jpg')
img = mpimg.imread(test_images[0])
pipeline = Pipeline(mtx, dist)
pipeline.set_warp_coordinates(img)
pipeline.process_image(img, './output_images/3.png', 2)

pipeline.prev_left_fit = []
pipeline.prev_right_fit = []

video_name = './project_video.mp4'
video_clip = VideoFileClip(video_name)
pipeline.set_warp_coordinates(video_clip.get_frame(0))
video_result = video_clip.fl_image(lambda xx: pipeline.pipeline(xx))
video_result.write_videofile('./result.mp4')
