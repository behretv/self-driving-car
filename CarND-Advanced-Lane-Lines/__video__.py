import time
import cv2
import glob
import numpy as np

from moviepy.video.io.VideoFileClip import VideoFileClip

from advanced_lane_lines import camera_calibration
from advanced_lane_lines.pipeline import Pipeline

images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

#################
# Load video data
video_name = './test_videos/project_video.mp4'
video_clip = VideoFileClip(video_name)

pipeline = Pipeline(mtx, dist)
pipeline.warp_coordinates(video_clip.get_frame(0))
video_result = video_clip.fl_image(pipeline.pipeline)
video_result.write_videofile('./output_videos/result.mp4')

data = np.array(pipeline.list_radius)
min_values = data.min(axis=0, initial=0)
max_values = data.max(axis=0, initial=0)
print(min_values)
print(max_values)

# Play video
time.sleep(1)
cap = cv2.VideoCapture("./output_videos/result_result.mp4")
ret, frame = cap.read()
while (1):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('frame', frame)
