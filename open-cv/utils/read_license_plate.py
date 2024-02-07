import os
import cv2
import pytesseract
import re


def read_license_plate(license_plate_crop):
    license_plate_text = extract_text(license_plate_crop)
    filtered_text, _ = filter_text(license_plate_text)

    return filtered_text, _


def extract_text(license_plate_crop):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Specify the path to tur.traineddata relative to the script
    tessdata_dir_config = f'--tessdata-dir "{script_dir}"'
    cv2.imshow("Frame", license_plate_crop)
    cv2.waitKey(1)

    # OCR with Tesseract
    license_plate_text = pytesseract.image_to_string(
        license_plate_crop,
        config=f"--psm 8 --oem 3 -l tur {tessdata_dir_config}",
    )

    return license_plate_text


def filter_text(license_plate_text):
    cleaned_text = re.sub("[^a-zA-Z0-9]", "", license_plate_text)

    # Additional filter: Check if the first two or three characters are alphanumeric
    if (
        cleaned_text[:2].isdigit()
        and 7 <= len(cleaned_text) <= 8
        and (cleaned_text[-3:].isdigit() or cleaned_text[-2:].isdigit())
    ):
        return cleaned_text, None
    else:
        return None, None
