import glob
import numpy as np

from moviepy.video.io.VideoFileClip import VideoFileClip

from advanced_lane_lines import camera_calibration
from advanced_lane_lines.pipeline import Pipeline

images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

#################
# Load video parameter
video_name = './test_videos/project_video.mp4'
video_clip = VideoFileClip(video_name)

pipeline = Pipeline(mtx, dist)
pipeline.warp_coordinates(video_clip.get_frame(0))
video_result = video_clip.fl_image(pipeline.pipeline)
video_result.write_videofile('./output_videos/result.mp4')

data = np.array(pipeline.lane.list_curvatures)
min_values = data.min(axis=0, initial=0)
max_values = data.max(axis=0, initial=0)
print(min_values)
print(max_values)
