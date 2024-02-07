import cv2
from ultralytics import YOLO
from sort.sort.sort import *
import pytesseract


def load_initials():
    # Configure Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

    # tracker
    mot_tracker = Sort()

    # load models
    coco_model = YOLO("./models/yolov8n.pt")
    license_plate_detector = YOLO("./models/license_plate_detector.pt")

    vehicles = [2, 3, 5, 7]

    return coco_model, license_plate_detector, mot_tracker, vehicles
