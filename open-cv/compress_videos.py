import os

from utils.compress_video import compress_video

video_directory = "./videos"

# Iterating through the directory
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):
        compress_video(filename)
