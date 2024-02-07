from main_dev import read_license
from utils.load_initials import load_initials
import os
import time

video_directory = "./compressed_videos"
coco_model, license_plate_detector, mot_tracker, vehicles = load_initials()
start_time = time.time()

for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):
        name_without_extension, _ = os.path.splitext(filename)
        read_license(
            f"{video_directory}/{filename}",
            f"./compressed_outputs/{name_without_extension}.csv",
            vehicles,
            coco_model,
            license_plate_detector,
            mot_tracker,
        )
# End the timer
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
