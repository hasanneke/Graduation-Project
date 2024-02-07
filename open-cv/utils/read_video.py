from ultralytics import YOLO
from utils.get_car import get_car
from utils.read_license_plate import read_license_plate
from sort.sort.sort import *
import numpy as np


def read_video(vehicles, cap, coco_model, license_plate_detector, mot_tracker):
    results = {}
    # read frames
    frame_count = 0
    ret = True

    # Calculate the interval for 6 frames per
    frame_interval = 6
    while ret:
        ret, frame = cap.read()
        print(ret and frame_count % frame_interval == 0)
        if ret and frame_count % frame_interval == 0:
            results[frame_count] = {}

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
                    license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

                    license_plate_text, _ = read_license_plate(license_plate_crop)

                    if license_plate_text:
                        results[frame_count][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "bbox": [x1, y1, x2, y2],
                                "text": license_plate_text,
                                "bbox_score": score,
                                "text_score": _,
                            },
                        }
        frame_count += 1

    return results
