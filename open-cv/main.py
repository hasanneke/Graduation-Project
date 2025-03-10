import numpy as np
import util
from ultralytics import YOLO
import cv2


from sort.sort.sort import *
from util import get_car, read_license_plate ,write_csv

results = {}
mot_tracker = Sort()


#Load model

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

#load video

cap = cv2.VideoCapture('./Sadece okunanlar(x0.8 hız).mp4')


vehicles = [2,3,5,7]
"""
    2:car
    3:motorbike
    5:bus
    7:truck
"""
#read freames
frame_number = -1
ret = True
# For 60 frame = 1 second
while ret:
    frame_number = frame_number+1
    ret, frame = cap.read()
    if ret:
        if frame_number > 7800:
            break
        results[frame_number] = {}
        #deteect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        # print(detections)

        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = detection

            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])

        #track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        #detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = license_plate
            #assign licesne plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # if frame_number % 20 ==0:
                #     cv2.imshow('threshold', license_plate_crop_thresh)
                #     cv2.waitKey(0)

                #read license plate number
                if score>0.4:
                    util.llmtest(license_plate_crop)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(license_plate_text)
                if license_plate_text is not None:
                    results[frame_number][car_id] = {'car':{'bbox':[xcar1, ycar1, xcar2, ycar2]},
                                                     'license_plate':{'bbox':[x1,y1,x2,y2],
                                                                      'text':license_plate_text,
                                                                      'bbox_score':score,
                                                                      'text_score':license_plate_text_score}}
                    # if license_plate_text_score > 0.2:
                    #     print(license_plate_text)
                    #     cv2.imshow('threshold', license_plate_crop_thresh)
                    #     cv2.waitKey(0)

#write results


write_csv(results,'./readedChanged.csv')
