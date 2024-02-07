from utils.compress_video import compress_video
from utils.load_initials import load_initials
from utils.read_video import read_video
from utils.write_csv import write_csv
import cv2


def read_license(
    video_path, output_path, vehicles, coco_model, license_plate_detector, mot_tracker
):
    # load video
    cap = cv2.VideoCapture(video_path)

    # read license plates
    results = read_video(vehicles, cap, coco_model, license_plate_detector, mot_tracker)

    # write to csv file
    write_csv(results, output_path)


def main():
    # load initials
    coco_model, license_plate_detector, mot_tracker, vehicles = load_initials()
    read_license(
        "./videos/sample1.mp4",
        "./results/sample1.csv",
        vehicles,
        coco_model,
        license_plate_detector,
        mot_tracker,
    )


if __name__ == "__main__":
    main()
