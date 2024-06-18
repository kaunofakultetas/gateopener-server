import os
import time
import cv2
import imagezmq
import json
import easyocr
import numpy as np





def deskew_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find contours in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # Calculate the angle of each line
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            angles.append(angle)
    
    # Compute the median angle of all lines
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0  # No skew detected
    
    # Rotate the image around its center to correct the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return corrected_img




def preprocess_image(img):
    img = deskew_image(img)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(9, 9))
    img = clahe.apply(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img





def detect():
    # Setup EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Setup ImageZMQ Server
    imageHub = imagezmq.ImageHub()

    # Iterate through all received frames
    while True:
        # Get new image frame
        (videoSenderName, img) = imageHub.recv_image()

        # Start the timer
        start_time = time.time()

        # Preprocess image
        # img = preprocess_image(img)

        # Run EasyOCR on the image
        results = reader.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        areas_and_texts = []
        for result in results:
            top_left, _, bottom_right, bottom_left = result[0]
            text = result[1]

            # Calculate the area of the bounding box
            width = np.linalg.norm(np.array(top_left) - np.array(bottom_right) / np.sqrt(2))  # Adjusted for rotation
            height = np.linalg.norm(np.array(top_left) - np.array(bottom_left) / np.sqrt(2))  # Adjusted for rotation
            area = width * height

            # Store area, text, and x-coordinate of top-left corner
            areas_and_texts.append((area, text, top_left[0]))

        # Proceed if there are detected texts
        if areas_and_texts:
            # Filter based on area threshold
            max_area = max(areas_and_texts, key=lambda x: x[0])[0]
            threshold_area = max_area * 0.5
            filtered_texts = [text for area, text, x_coord in areas_and_texts if area > threshold_area]

            # Sort filtered texts by their x-coordinate (horizontal position)
            sorted_texts = sorted([(x_coord, text) for area, text, x_coord in areas_and_texts if area > threshold_area], key=lambda x: x[0])

            # Concatenate sorted texts
            possibleNumberplate = ''.join([text for _, text in sorted_texts])
        else:
            possibleNumberplate = ''

        # possibleNumberplate = possibleNumberplate.replace(" ", "").upper()
        imageHub.send_reply(json.dumps({'detection': possibleNumberplate}, indent=4).encode())
        print(f"Proc. Time: {int((time.time() - start_time) * 1000)} ms")

detect()
