import string

import cv2
import easyocr
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import functools

# Initialize the OCR reader
reader = easyocr.Reader(['tr'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

def cut_first_char(text):
    if len(text)>8:
        new_text = text[1:]
        return new_text
    else:
        return text

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # if len(text) != 7:
    #     return False
    #
    # if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
    #    (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
    #    (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
    #    (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
    #    (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
    #    (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
    #    (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
    #     return True
    # else:
    #     return False
    """For Turkish License Plate Format"""
    if len(text) != 7 and len(text) != 8:
        return False


    if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
        #34 ABC 34
        return True
    elif (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
         (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
         (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
         (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
         (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
         (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        #34 TL 3666
        return True
    elif (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
         (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
         (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
         (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
         (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
         (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        #34 TLK 366
        return True
    elif (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
         (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
         (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
         (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
         (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
        #34 TL 366
        return True
    elif (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
         (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
         (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
         (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
         (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
         (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
        #34 T 3666
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    """For sample Video"""
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_
"""For Turkish License Plate Format"""
    # license_plate_ = ''
    # mapping = {0: dict_char_to_int, 1: dict_char_to_int, 4: dict_int_to_char, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int,
    #            2: dict_int_to_char, 3: dict_int_to_char}
    # for j in [0, 1, 2, 3, 4, 5, 6]:
    #     if text[j] in mapping[j].keys():
    #         license_plate_ += mapping[j][text[j]]
    #     else:
    #         license_plate_ += text[j]
    #
    # return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    text=""
    score=""
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        # if len(text)>8:
        #     text = cut_first_char(text)
        text = text.upper().replace(' ', '')
        # if license_complies_format(text):
        #     # return format_license(text), score
        #       return text,score

    return text, score


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
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


def llmtest(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring and thresholding
    # to reveal the characters on the license plate
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

    # Perform connected components analysis on the thresholded images and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 70  # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 20  # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the bounding boxes from left to right, top to bottom
    # sort by Y first, and then sort by X if Ys are similar
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    ###############################################
    # The second stage
    ###############################################

    # Define constants
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128

    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    # Load the pre-trained convolutional neural network
    model = load_model("./characters_model.weights", compile=False)

    vehicle_plate = ""
    # Loop over the bounding boxes
    for rect in boundingBoxes:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = mask[y:y + h, x:x + w]
        crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        # Prepare data for prediction
        crop = crop.astype("float") / 255.0
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)

        # Make prediction
        prob = model.predict(crop)[0]
        idx = np.argsort(prob)[-1]
        vehicle_plate += chars[idx]

        # Show bounding box and prediction on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, chars[idx], (x, y + 15), 0, 0.8, (0, 0, 255), 2)

    # Show final image
    #cv2.imshow('Final', image)
    print("Vehicle plate: " + vehicle_plate)
    cv2.waitKey(0)