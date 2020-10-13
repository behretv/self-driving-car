import matplotlib.image as mpimg
import glob

from moviepy.video.io.VideoFileClip import VideoFileClip

from advanced_lane_lines import camera_calibration
from advanced_lane_lines.pipeline import Pipeline


images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

test_images = glob.glob('./test_images/*.jpg')
pipeline = Pipeline(mtx, dist)

for i in range(0, 5):
    img = mpimg.imread(test_images[i])
    pipeline.warp_coordinates(img)
    pipeline.process_image(img, './output_images/{}.png'.format(i), i)
    pipeline.process_image(img, './output_images/{}.png'.format(i), i)

pipeline.prev_left_fit = []
pipeline.prev_right_fit = []
video_name = './test_videos/project_video.mp4'
video_clip = VideoFileClip(video_name).subclip(0, 1)
pipeline.warp_coordinates(video_clip.get_frame(0))
video_result = video_clip.fl_image(pipeline.pipeline)
video_result.write_videofile('./output_videos/short_result.mp4')
