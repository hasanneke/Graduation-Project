import cv2
import pytesseract
from ultralytics import YOLO
from sort.sort.sort import *
import numpy as np
import re

results = {}

mot_tracker = Sort()


# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./license_plate_detector.pt")

# load video
cap = cv2.VideoCapture("./sample4.mp4")

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Specify the path to tur.traineddata relative to the script
tessdata_dir_config = f'--tessdata-dir "{script_dir}"'


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def write_csv(results, output_path):
    with open(output_path, "w") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            )
        )

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if (
                    "car" in results[frame_nmr][car_id].keys()
                    and "license_plate" in results[frame_nmr][car_id].keys()
                    and "text" in results[frame_nmr][car_id]["license_plate"].keys()
                ):
                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["car"]["bbox"][0],
                                results[frame_nmr][car_id]["car"]["bbox"][1],
                                results[frame_nmr][car_id]["car"]["bbox"][2],
                                results[frame_nmr][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_nmr][car_id]["license_plate"]["bbox"][0],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][1],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][2],
                                results[frame_nmr][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_nmr][car_id]["license_plate"]["bbox_score"],
                            results[frame_nmr][car_id]["license_plate"]["text"],
                            results[frame_nmr][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()


def read_license_plate(license_plate_crop):
    # # Convert to grayscale
    # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # # Thresholding
    # _, license_plate_crop_thresh = cv2.threshold(
    #     license_plate_crop_gray, 120, 255, cv2.THRESH_BINARY_INV
    # )
    cv2.imshow("Frame", license_plate_crop)
    cv2.waitKey(1)
    # OCR with Tesseract
    license_plate_text = pytesseract.image_to_string(
        license_plate_crop,
        config=f"--psm 8 --oem 3 -l tur {tessdata_dir_config}",
    )
    cleaned_text = re.sub("[^a-zA-Z0-9]", "", license_plate_text)
    if 7 <= len(cleaned_text) <= 8:
        return cleaned_text, None

    return None, None


while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr < 100:
        results[frame_nmr] = {}

        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
                # cv2.imshow("Frame", license_plate_crop)
                # cv2.waitKey(1)
                # read license plate number
                license_plate_text, _ = read_license_plate(license_plate_crop)

                if license_plate_text:
                    results[frame_nmr][car_id] = {
                        "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": license_plate_text,
                            "bbox_score": score,
                            "text_score": _,
                        },
                    }

# write results
write_csv(results, "./sample.csv")
