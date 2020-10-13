import matplotlib.image as mpimg
import glob

from advanced_lane_lines import camera_calibration
from advanced_lane_lines.pipeline import Pipeline

# Calibration
images = glob.glob('./camera_cal/calibration*.jpg')
mtx, dist, img = camera_calibration(images)

# Process images
test_images = glob.glob('./test_images/*.jpg')
pipeline_img = Pipeline(mtx, dist)

for i in range(0, 6):
    img = mpimg.imread(test_images[i])
    pipeline_img.warp_coordinates(img)
    pipeline_img.process_image(img, './output_images/{}.png'.format(i), i)
    pipeline_img.process_image(img, './output_images/{}.png'.format(i), i)

