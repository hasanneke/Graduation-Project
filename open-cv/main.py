from utils.load_initials import load_initials
from utils.read_video import read_video
from utils.write_csv import write_csv


def main():
    # load initials
    coco_model, license_plate_detector, cap, mot_tracker, vehicles = load_initials(
        video_path="./videos/sample1.mp4"
    )

    # read license plates
    results = read_video(vehicles, cap, coco_model, license_plate_detector, mot_tracker)

    # write to csv file
    write_csv(results, "./results/sample1.csv")


if __name__ == "__main__":
    main()
