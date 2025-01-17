import os
import json
import cv2
from flask_opencv_streamer.streamer import Streamer
from threading import Thread, Lock
from datetime import datetime, date, timedelta
import requests
import collections
import hashlib
import time
import numpy as np
import imagezmq
import zmq


# ImageZMQ tutorial:
# https://pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/









# +--------------------------------------------------+
# +----------- Camera Video Stream Puller -----------+
# +--------------------------------------------------+
# ENV's
INPUT_CAMERA_STREAM_URL = os.getenv('INPUT_CAMERA_STREAM_URL')

cap = cv2.VideoCapture(INPUT_CAMERA_STREAM_URL)
CAMERA_STREAM_FRAMERATE = int(cap.get(cv2.CAP_PROP_FPS))
def cameraVideoPuller_getNext():
    ret, frame = cap.read()
    return frame

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+









# +--------------------------------------------------+
# +--------------- Frame Skipper --------------------+
# +--------------------------------------------------+
# ENV's
PROCESS_EVERY_N_TH_FRAME = int(os.getenv('PROCESS_EVERY_N_TH_FRAME', 1))

frameSkipperCounter = 0
def frameSkipper_iter():
    global frameSkipperCounter
    frameSkipperCounter += 1
    if(frameSkipperCounter % PROCESS_EVERY_N_TH_FRAME != 0):
        return True

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+










# +--------------------------------------------------+
# +--------- Detections video file writer -----------+
# +--------------------------------------------------+
# ENV's
WRITE_DET_TO_VIDEO_FILE_FILENAMESTART = os.getenv('WRITE_DET_TO_VIDEO_FILE_FILENAMESTART', None)
WRITE_DET_SECONDS_BEFORE_TRIGGER = int(os.getenv('WRITE_DET_SECONDS_BEFORE_TRIGGER', 7))
WRITE_DET_TO_VIDEO_FILE_VERBOSITY = int(os.getenv('WRITE_DET_TO_VIDEO_FILE_VERBOSITY', 0))
WRITE_DET_FULL_QUALITY = int(os.getenv('WRITE_DET_FULL_QUALITY', 0))


video_input_resolution = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
video_input_pixel_count = video_input_resolution[0] * video_input_resolution[1]

det_video_resolution = (video_input_resolution[0], video_input_resolution[1])
det_video_filename, det_video_fourcc, det_video_out = None, None, None
detVideoFramesDeque = collections.deque(maxlen=int((CAMERA_STREAM_FRAMERATE*WRITE_DET_SECONDS_BEFORE_TRIGGER)/PROCESS_EVERY_N_TH_FRAME))



# Delete old and empty video files
timestamp_now = time.time()
timestamp_days_ago = timestamp_now - 14 * 86400  # X days in seconds
if WRITE_DET_TO_VIDEO_FILE_FILENAMESTART is not None:
    for filename in os.listdir('./saved_videos'):
        if filename.startswith(WRITE_DET_TO_VIDEO_FILE_FILENAMESTART):
            file_path = os.path.join('./saved_videos', filename)
            if os.path.isfile(file_path):
                file_modification_time = os.path.getmtime(file_path)  # Use modification time
                
                # Check if the file is older than X days
                if file_modification_time < timestamp_days_ago:
                    print(f"Deleting old file: {file_path}")
                    os.remove(file_path)
                
                # Delete empty files
                elif os.path.getsize(file_path) < 1*1024*1024:  # Less than 1MB
                    print(f"Deleting empty file: {file_path}")
                    os.remove(file_path)




if(WRITE_DET_TO_VIDEO_FILE_FILENAMESTART is not None):
    timeNow = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    det_video_filename = f'./saved_videos/{WRITE_DET_TO_VIDEO_FILE_FILENAMESTART}_{timeNow}.avi'
    det_video_fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not WRITE_DET_FULL_QUALITY:
        new_height = 720
        aspect_ratio = video_input_resolution[0] / video_input_resolution[1]
        new_width = int(new_height * aspect_ratio)
        det_video_resolution = (new_width, new_height)

    det_video_out = cv2.VideoWriter(det_video_filename, det_video_fourcc, int(CAMERA_STREAM_FRAMERATE/PROCESS_EVERY_N_TH_FRAME), det_video_resolution)



def detVideoWriterToFile_iter(frame, triggerOpen, triggerDetect):
    if(WRITE_DET_TO_VIDEO_FILE_FILENAMESTART is not None):
        frame_tmp = frame.copy()
        
        if not WRITE_DET_FULL_QUALITY:
            frame_tmp = cv2.resize(frame_tmp, det_video_resolution)
        detVideoFramesDeque.append(frame_tmp)
        
        dumpDeque = False
        if(triggerOpen):
            dumpDeque = True
        elif(triggerDetect and WRITE_DET_TO_VIDEO_FILE_VERBOSITY == 1):
            dumpDeque = True

        if(dumpDeque):
            for videoFrameFromDeque in list(detVideoFramesDeque):
                detVideoFramesDeque.popleft()
                det_video_out.write(videoFrameFromDeque)

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+











# +--------------------------------------------------+
# +------------- Raw video file writer --------------+
# +--------------------------------------------------+
# ENV's
WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART = os.getenv('WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART', None)
WRITE_RAW_SECONDS_BEFORE_TRIGGER = int(os.getenv('WRITE_RAW_SECONDS_BEFORE_TRIGGER', 7))
WRITE_RAW_TO_VIDEO_FILE_VERBOSITY = int(os.getenv('WRITE_RAW_TO_VIDEO_FILE_VERBOSITY', 0))
WRITE_RAW_FULL_QUALITY = int(os.getenv('WRITE_RAW_FULL_QUALITY', 0))



video_input_resolution = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
raw_out_resolution = (video_input_resolution[0], video_input_resolution[1])
raw_video_filename, raw_video_fourcc, raw_video_out = None, None, None
rawVideoFramesDeque = collections.deque(maxlen=int((CAMERA_STREAM_FRAMERATE*WRITE_RAW_SECONDS_BEFORE_TRIGGER)/PROCESS_EVERY_N_TH_FRAME))




# Delete old and empty video files
timestamp_now = time.time()
timestamp_days_ago = timestamp_now - 14 * 86400  # X days in seconds
if WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART is not None:
    for filename in os.listdir('./saved_videos'):
        if filename.startswith(WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART):
            file_path = os.path.join('./saved_videos', filename)
            if os.path.isfile(file_path):
                file_modification_time = os.path.getmtime(file_path)  # Use modification time
                
                # Check if the file is older than X days
                if file_modification_time < timestamp_days_ago:
                    print(f"Deleting old file: {file_path}")
                    os.remove(file_path)
                
                # Delete empty files
                elif os.path.getsize(file_path) < 1*1024*1024:  # Less than 1MB
                    print(f"Deleting empty file: {file_path}")
                    os.remove(file_path)




if(WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART is not None):
    timeNow = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    raw_video_filename = f'./saved_videos/{WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART}_{timeNow}.avi'
    raw_video_fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not WRITE_RAW_FULL_QUALITY:
        new_height = 720
        aspect_ratio = video_input_resolution[0] / video_input_resolution[1]
        new_width = int(new_height * aspect_ratio)
        raw_out_resolution = (new_width, new_height)

    raw_video_out = cv2.VideoWriter(raw_video_filename, raw_video_fourcc, int(CAMERA_STREAM_FRAMERATE/PROCESS_EVERY_N_TH_FRAME), raw_out_resolution)



def rawVideoWriterToFile_iter(frame, triggerOpen, triggerDetect):
    if(WRITE_RAW_TO_VIDEO_FILE_FILENAMESTART is not None):
        frame_tmp = frame.copy()
        
        if not WRITE_RAW_FULL_QUALITY:
            frame_tmp = cv2.resize(frame_tmp, raw_out_resolution)
        rawVideoFramesDeque.append(frame_tmp)
        
        dumpDeque = False
        if(triggerOpen):
            dumpDeque = True
        elif(triggerDetect and WRITE_RAW_TO_VIDEO_FILE_VERBOSITY == 1):
            dumpDeque = True

        if(dumpDeque):
            for videoFrameFromDeque in list(rawVideoFramesDeque):
                rawVideoFramesDeque.popleft()
                raw_video_out.write(videoFrameFromDeque)

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+














# +--------------------------------------------------+
# +------------ Live Detections Streamer ------------+
# +--------------------------------------------------+
# ENV's
STREAM_FULL_QUALITY = int(os.getenv('STREAM_FULL_QUALITY', 0))


streamer = Streamer(3030, False)
streamer.start_streaming()



stream_resolution = (video_input_resolution[0], video_input_resolution[1])
if not STREAM_FULL_QUALITY:
    new_height = 720
    aspect_ratio = video_input_resolution[0] / video_input_resolution[1]
    new_width = int(new_height * aspect_ratio)
    stream_resolution = (new_width, new_height)


# Stream video through flask streamer
def liveDetectionsStreamer_iter(frame):
    frame_tmp = frame.copy()

    if(not STREAM_FULL_QUALITY):
        frame_tmp = cv2.resize(frame_tmp, stream_resolution)

    streamer.update_frame(frame_tmp)

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+



















# +--------------------------------------------------+
# +----------- Neural Objects detector --------------+
# +--------------------------------------------------+
# ENV's
MODULE_OPEN_ZONES =  os.getenv('MODULE_OPEN_ZONES', "false").lower() == "true"
NEURAL_OBJECTS_HOST_PORT = os.getenv('NEURAL_OBJECTS_HOST_PORT', "neural-objects:5555")
OPEN_BOX_POSITIONS = json.loads(os.getenv('OPEN_BOX_POSITIONS')) if os.getenv('OPEN_BOX_POSITIONS') else None





# Create an ImageSender object
if(MODULE_OPEN_ZONES):
    sender = imagezmq.ImageSender(connect_to=f"tcp://{NEURAL_OBJECTS_HOST_PORT}")



# Send frame for processing
def neuralObjectsDetector_sendFrame(frame):
    response = sender.send_image("camera1", frame)
    sender.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    sender.zmq_socket.setsockopt(zmq.RCVTIMEO, 800 )  # will raise a ZMQError exception after x ms
    sender.zmq_socket.setsockopt(zmq.SNDTIMEO, 800 )  # will raise a ZMQError exception after x ms

    # Received information from neural processing
    responseJson = json.loads(response.decode('utf-8'))
    # print(json.dumps(responseJson, indent=4))
    return responseJson


# Lets check if object center is in the opener box
def gate_opener_checkObjectInOpenPos(centerCoordinate):
    for openBoxPosition in OPEN_BOX_POSITIONS:
        if(centerCoordinate[0] > openBoxPosition[0] and centerCoordinate[0] < openBoxPosition[2] and
        centerCoordinate[1] > openBoxPosition[1] and centerCoordinate[1] < openBoxPosition[3]):
            return True


# Plot Opener Boxes on the screen
def gate_opener_drawOpenBoxPositions(frame):
    for openBoxPositionID in range(len(OPEN_BOX_POSITIONS)):
        plot_one_box(OPEN_BOX_POSITIONS[openBoxPositionID], frame, label="Open Box " + str(openBoxPositionID + 1), color=(255, 0, 0), text_color=(255, 255, 255), line_thickness=1)
    return frame


# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+


















# +--------------------------------------------------+
# +------- Neural Numberplate detector/reader -------+
# +--------------------------------------------------+
# ENV's
MODULE_NUMBERPLATE_READER =  os.getenv('MODULE_NUMBERPLATE_READER', "false").lower() == "true"

NEURAL_NP_DETECTOR_HOST_PORT = os.getenv('NEURAL_NP_DETECTOR_HOST_PORT', 'neural-np-location:5555')
NEURAL_NP_READER_HOST_PORT = os.getenv('NEURAL_NP_READER_HOST_PORT', 'neural-np-easyocr:5555')
ZOOMED_IN_BOX = json.loads(os.getenv('ZOOMED_IN_BOX')) if os.getenv('ZOOMED_IN_BOX') else None


ALLOWED_NUMBERPLATES_API = os.getenv('ALLOWED_NUMBERPLATES_API')



allowedNumberPlates = {}

if NEURAL_NP_DETECTOR_HOST_PORT is None     and MODULE_NUMBERPLATE_READER == True:
    print("[*] Error: Define env variable 'NEURAL_NP_DETECTOR_HOST_PORT' for example:  host:port ")
    exit(1)
if NEURAL_NP_READER_HOST_PORT is None       and MODULE_NUMBERPLATE_READER == True:
    print("[*] Error: Define env variable 'NEURAL_NP_READER_HOST_PORT' for example:  host:port ")
    exit(1)

# Create an ImageSender object
if(MODULE_NUMBERPLATE_READER):
    senderNpDetector = imagezmq.ImageSender(connect_to=f"tcp://{NEURAL_NP_DETECTOR_HOST_PORT}")
    senderNpReader = imagezmq.ImageSender(connect_to=f"tcp://{NEURAL_NP_READER_HOST_PORT}")
    allowedNumberPlates = json.loads(requests.get(ALLOWED_NUMBERPLATES_API).text)




# Send frame for processing
def neuralLPDetector_sendFrame(frame):
    response = senderNpDetector.send_image("camera1", frame)
    senderNpDetector.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    senderNpDetector.zmq_socket.setsockopt(zmq.RCVTIMEO, 500 )  # will raise a ZMQError exception after x ms
    senderNpDetector.zmq_socket.setsockopt(zmq.SNDTIMEO, 500 )  # will raise a ZMQError exception after x ms

    # Received information from neural processing
    responseJson = json.loads(response.decode('utf-8'))
    # print(json.dumps(responseJson, indent=4))
    return responseJson



# Send frame for processing
def neuralLPReader_sendFrame(frame):
    response = senderNpReader.send_image("camera1", frame)
    senderNpReader.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    senderNpReader.zmq_socket.setsockopt(zmq.RCVTIMEO, 500 )  # will raise a ZMQError exception after x ms
    senderNpReader.zmq_socket.setsockopt(zmq.SNDTIMEO, 500 )  # will raise a ZMQError exception after x ms

    # Received information from neural processing
    responseJson = json.loads(response.decode('utf-8'))
    # print(json.dumps(responseJson, indent=4))
    return responseJson





def processNumberplateDetections(frame, NP_detectionsJson, detectionOffset=[0,0] ):
    capturedNumberplate = ""
    triggerGateOpenSignal = False
    somethingDetected = False
    biggestDetectionPixelCount = 0


    for detectionJson in NP_detectionsJson:
        detectionName, detectionConfidence, detectionBox = detectionJson['class'], detectionJson['confidence'], detectionJson['bbox']
        x1, y1 = detectionBox['x1'] + detectionOffset[0], detectionBox['y1'] + detectionOffset[1]
        x2, y2 = detectionBox['x2'] + detectionOffset[0], detectionBox['y2'] + detectionOffset[1]

        if(biggestDetectionPixelCount < (x2-x1)*(y2-y1)):
            biggestDetectionPixelCount = (x2-x1)*(y2-y1)


        numberplateReading = ""
        if( detectionName in  ["LP", "number-plate"] ):
            # Crop numberplate image out of frame
            cropped_np_image = frame[y1:y2, x1:x2]
            
            # Save to directory the numberplate
            timeNow = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            md5_hash = hashlib.md5(cropped_np_image.tobytes()).hexdigest()
            cv2.imwrite(f'./saved_numberplates/{timeNow}_{md5_hash[:16]}.jpg', cropped_np_image)

            # Preprocess cropped numberplate image
            # cropped_np_image = preprocess_image(cropped_np_image)

            # Send cropped numberplate for OCR reading
            NP_detectionTextJson = neuralLPReader_sendFrame(cropped_np_image)

            # V1
            # numberplateReading = NP_detectionTextJson["detection"] 

            # V2
            sorted_detections = sorted(NP_detectionTextJson, key=lambda char: char['bbox']['x1'])
            numberplateReading = ''.join(char['class'] for char in sorted_detections)
            


            
            # print(json.dumps(NP_detectionTextJson, indent=4))
            
            if(len(numberplateReading) > 2):
                # Get possible numberplate texts
                possibleNumberplates = generatePossibleNumberplates(numberplateReading)

                # Iterate through all possible numberplates and open the gate
                for possibleNumberplate in possibleNumberplates:
                    if(possibleNumberplate in allowedNumberPlates):
                        triggerGateOpenSignal = True
                        capturedNumberplate = possibleNumberplate
                        print(f"[*] Sent opening signal by reading: {numberplateReading}, correction trigered: {possibleNumberplate}")
                        break
 
        # Plot boxes around all detections
        label = f'{detectionName} {detectionConfidence:.2f}'
        if(numberplateReading != ""):
            label = f"{label} ({numberplateReading})"
        plot_one_box([x1, y1, x2, y2], frame, color=(0, 255, 0), label=label, line_thickness=3)
        somethingDetected = True


    # Zoomed In box on the screen
    if(ZOOMED_IN_BOX is not None):
        plot_one_box(ZOOMED_IN_BOX, frame, label="Zoomed IN Box", color=(255, 0, 0), text_color=(255, 255, 255), line_thickness=3)           

    return frame, capturedNumberplate, triggerGateOpenSignal, somethingDetected, biggestDetectionPixelCount
# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+











# +--------------------------------------------------+
# +-------------- Open Request Sender ---------------+
# +--------------------------------------------------+
# Env's
MODULE_ENABLED_GATE_OPENER = True # Gate Opener API (Raspberry Pi)
OPENER_REQUEST_LIMITER_SECONDS = int(os.getenv('OPENER_REQUEST_LIMITER_SECONDS', 5))

OPENER_REQUEST_URL = os.getenv('OPENER_REQUEST_URL', None)


STATUS_LABEL_POS =      json.loads(os.getenv('STATUS_LABEL_POS')) if os.getenv('STATUS_LABEL_POS') else [100, 200]
STATUS_LABEL_OPEN =     os.getenv('STATUS_LABELS', "OPENED,CLOSED,DISABLED").split(',')[0]
STATUS_LABEL_CLOSED =   os.getenv('STATUS_LABELS', "OPENED,CLOSED,DISABLED").split(',')[1]
STATUS_LABEL_DISABLED = os.getenv('STATUS_LABELS', "OPENED,CLOSED,DISABLED").split(',')[2]


CLOCK_ENABLED =         os.getenv('CLOCK_ENABLED', "false").lower() == "true"
CLOCK_LABEL_POS =       json.loads(os.getenv('STATUS_LABEL_POS')) if os.getenv('STATUS_LABEL_POS') else [100, 130]


OPENER_REQUEST_LIMITER_FRAME_COUNT = int((OPENER_REQUEST_LIMITER_SECONDS * CAMERA_STREAM_FRAMERATE)/PROCESS_EVERY_N_TH_FRAME)


if OPENER_REQUEST_URL is None:
    print("[*] Error: Define env variable 'OPENER_REQUEST_URL' for example:  http://host:port/example/path ")
    exit(1)



openRequestLimiter = 0
def openRequestSender_iter(frame, triggerGateOpenSignal, capturedNumberplate=""):
    global openRequestLimiter

    # Status Label and request sender to Rasp Pi
    backgroundColor = [0, 0, 255] # Red
    label = STATUS_LABEL_CLOSED


    if(openRequestLimiter > 0):
        openRequestLimiter -= 1

    
    if(MODULE_ENABLED_GATE_OPENER):

        if(triggerGateOpenSignal):
            backgroundColor = [0, 255, 0] # Green
            label = STATUS_LABEL_OPEN

            if(openRequestLimiter == 0):
                openRequestLimiter = OPENER_REQUEST_LIMITER_FRAME_COUNT
                requests.get(OPENER_REQUEST_URL + "?numberplate=" + capturedNumberplate, timeout=1)
        
        elif(openRequestLimiter > 0):
            backgroundColor = [0, 165, 255] # Orange
            label = STATUS_LABEL_OPEN

    else:
        backgroundColor = [0, 0, 0] # Black
        label = STATUS_LABEL_DISABLED
            


    # Print status labels on the screen
    tl = 3
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = STATUS_LABEL_POS[0] + t_size[0], STATUS_LABEL_POS[1] - t_size[1] - 3
    cv2.rectangle(frame, STATUS_LABEL_POS, c2, backgroundColor, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (STATUS_LABEL_POS[0], STATUS_LABEL_POS[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return frame


def printClockOnScreen(frame):
    if(CLOCK_ENABLED):
        clock_size = 3
        backgroundColor = [0, 0, 0] # Black
        label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tl = 2
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=(tl / 3)*clock_size, thickness=tf-1)[0]
        c2 = CLOCK_LABEL_POS[0] + t_size[0], CLOCK_LABEL_POS[1] - t_size[1] - 3
        cv2.rectangle(frame, CLOCK_LABEL_POS, c2, backgroundColor, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (CLOCK_LABEL_POS[0], CLOCK_LABEL_POS[1] - 2), 0, (tl / 3)*clock_size, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return frame


# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+










# +--------------------------------------------------+
# +------------------ Antifreezer -------------------+
# +--------------------------------------------------+
# ENV's
FREEZE_TIMEOUT_SECONDS = int(os.getenv('FREEZE_TIMEOUT_SECONDS', 3))


antifreezeFrameCounter = 0
def antifreezer_background_task():
    global antifreezeFrameCounter, FREEZE_TIMEOUT_SECONDS
    while True:
        time.sleep(FREEZE_TIMEOUT_SECONDS)

        if(antifreezeFrameCounter == 0):        # If there was no video frames in the last period
            os._exit(1)

        antifreezeFrameCounter = 0

daemon = Thread(target=antifreezer_background_task, daemon=True, name='Antifreezer')
daemon.start()


# Iterate through every frame that is received from camera
def antifreezer_iter():
    global antifreezeFrameCounter
    antifreezeFrameCounter += 1

# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+














# +--------------------------------------------------+
# +-------------------- Utilities -------------------+
# +--------------------------------------------------+

# Plots one bounding box on image img
def plot_one_box(x, img, color=(255, 0, 0), text_color=(0,0,0), label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=(tl / 3), thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, (tl / 3), text_color, thickness=tf, lineType=cv2.LINE_AA)



def deskew_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find contours in the image
    edges = cv2.Canny(gray, 50, 110, apertureSize=5)
    
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




def resize_image_to_specific_width(img, target_width=400):
    scale_factor = target_width / img.shape[1]
    new_height = int(img.shape[0] * scale_factor)
    resized_img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img



def preprocess_image(img):
    img = resize_image_to_specific_width(img)
    img = deskew_image(img)

    # Check the average brightness of the image
    mean_brightness = int(np.mean(img))

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object (Arguments are optional)
    def dynamic_clahe(img):
        # Analyze the image's contrast
        contrast = int(np.std(img))
        # print(contrast)
        
        # Adjust CLAHE parameters based on contrast
        if contrast < 30:
            clipLimit = 1.0
            tileGridSize = (5, 5)
        else:
            clipLimit = 1.0
            tileGridSize = (3, 3)
        
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(img)
    img = dynamic_clahe(img)


    if mean_brightness > 225 or mean_brightness < 90:
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply a median blur to reduce noise while preserving edges
    img = cv2.medianBlur(img, 3)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img




def generatePossibleNumberplates(numberplateReading):
    # Manual Correction
    possibleNumberplates = [numberplateReading]


    if(len(numberplateReading) == 7):
        possibleNumberplates.append(numberplateReading[0:6])
        possibleNumberplates.append(numberplateReading[1:7])


    if(len(numberplateReading) == 8):
        possibleNumberplates.append(numberplateReading[0:6])
        possibleNumberplates.append(numberplateReading[1:7])
        possibleNumberplates.append(numberplateReading[2:8])


    while True:
        initialCount = len(possibleNumberplates)

        for possibleNumberplate in possibleNumberplates:
            numberplateLetters = possibleNumberplate[0:3]
            numberplateNumbers = possibleNumberplate[3:6]

            # Fix letters side
            for replacement in [["0", "O"], ["0", "D"], ["1", "I"], ["2", "Z"], ["4", "A"], ["7", "T"], ["7", "Z"], ["8", "B"], ["8", "M"], ["W", "A"]]:
                modNumberplateLetters = numberplateLetters.replace(replacement[0], replacement[1])
                if(modNumberplateLetters + numberplateNumbers not in possibleNumberplates):
                    possibleNumberplates.append(modNumberplateLetters + numberplateNumbers)

            # Fix numbers side
            for replacement in [["A", "4"], ["J", "3"], ["7", "1"], ["S", "5"], ["8", "6"], ["S", "9"]]:
                modNumberplateNumbers = numberplateNumbers.replace(replacement[0], replacement[1])
                if(numberplateLetters + modNumberplateNumbers not in possibleNumberplates):
                    possibleNumberplates.append(numberplateLetters + modNumberplateNumbers)

            
        if(initialCount == len(possibleNumberplates)):
            break

    return possibleNumberplates


# +--------------------------------------------------+
# +--------------------------------------------------+
# +--------------------------------------------------+













updateAllowedFrameCounter = 0
while True:

    # Get next frame from camera
    frame = cameraVideoPuller_getNext()
    

    # Kill process if antifreeze function is no loger triggered
    antifreezer_iter()


    # Process every n'th frame depending on the required framerate
    if(frameSkipper_iter()):
        continue

    raw_frame = frame.copy()
    somethingDetected = False
    biggestDetectionPixelCount = 0

    # Send frame for processing (search numberplate location)
    
    if(MODULE_NUMBERPLATE_READER):

        # Update allowed numberplate list every 5 minutes
        updateAllowedFrameCounter += 1
        if(updateAllowedFrameCounter % int((300 * CAMERA_STREAM_FRAMERATE)/PROCESS_EVERY_N_TH_FRAME) == 0):
            try:
                allowedNumberPlates = json.loads(requests.get(ALLOWED_NUMBERPLATES_API).text)
            except:
                pass

        # Send frame to search for numberplates
        NP_detectionsJson = neuralLPDetector_sendFrame(frame)


        # If no detections on the normal frame - check zoomed in frame
        detectionOffset = [0, 0]
        if(ZOOMED_IN_BOX is not None):
            if(len(NP_detectionsJson) == 0):
                detectionOffset = [ZOOMED_IN_BOX[0], ZOOMED_IN_BOX[1]]
                x1, y1, x2, y2 = ZOOMED_IN_BOX

                # Crop zoomed in box out of frame and send to search for numberplates in it
                cropped_frame = frame[y1:y2, x1:x2]
                NP_detectionsJson = neuralLPDetector_sendFrame(cropped_frame)


        # Check all numberplate detections in the frame
        frame, capturedNumberplate, triggerGateOpenSignal, somethingDetected, biggestDetectionPixelCount = processNumberplateDetections(frame, NP_detectionsJson, detectionOffset)
   
        # Send signal to Raspberry pi and print status labels
        frame = openRequestSender_iter(frame, triggerGateOpenSignal, capturedNumberplate)

    

    # Send frame for processing (object detection for opening zones)
    elif(MODULE_OPEN_ZONES):
        triggerGateOpenSignal = False

        # Get object detections and iterate through every detection
        for detectionJson in neuralObjectsDetector_sendFrame(frame):
            detectionName, detectionConfidence, detectionBox = detectionJson['class'], detectionJson['confidence'], detectionJson['bbox']
            x1, y1, x2, y2 = detectionBox['x1'], detectionBox['y1'], detectionBox['x2'], detectionBox['y2']
            
            # Plot boxes around all detections
            label = f'{detectionName} {detectionConfidence:.2f}'
            plot_one_box([x1, y1, x2, y2], frame, color=(0, 255, 0), label=label)

            
            if(detectionName in ['car', 'truck', 'motorcycle', 'bus']):
                size_xy = (x2 - x1, y2 - y1)

                if((size_xy[0] > 200 and size_xy[1] > 100) or detectionName == 'motorcycle'):
                    centerCoordinate = (int((x1 + x2)/2), int((y1 + y2)/2))

                    # Draw center cross and circle in the middle
                    cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 0))
                    cv2.line(frame, (x1, y2), (x2, y1), color=(0, 255, 0))
                    cv2.circle(frame, centerCoordinate, 10, color=(0, 255, 0), thickness=2)

                    # Lets check if object center is in the opener box
                    triggerGateOpenSignal = gate_opener_checkObjectInOpenPos(centerCoordinate)
                    if(triggerGateOpenSignal):
                        break


        gate_opener_drawOpenBoxPositions(frame)

        # Send signal to Raspberry pi and print status labels
        frame = openRequestSender_iter(frame, triggerGateOpenSignal)


    # Clock Writer on the screen   
    frame = printClockOnScreen(frame)


    # Write detection results to file
    biggestDetectionFractionOfScreen = biggestDetectionPixelCount*1.0/video_input_pixel_count
    detVideoWriterToFile_iter(frame, openRequestLimiter > 0 or biggestDetectionFractionOfScreen > 0.02, somethingDetected)
    rawVideoWriterToFile_iter(raw_frame, openRequestLimiter > 0 or biggestDetectionFractionOfScreen > 0.02, somethingDetected)
    

    # Stream video through flask streamer
    liveDetectionsStreamer_iter(frame)

