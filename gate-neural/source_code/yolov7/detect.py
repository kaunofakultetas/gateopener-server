import os
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel

import imagezmq
from datetime import datetime, date, timedelta
import hashlib
import json



def detect():
    weights = os.getenv('DETECT_WEIGHTS')
    tracedModelPath = os.getenv('DETECT_TRACED_MODEL_PATH')
    imgsz = int(os.getenv('DETECT_IMGSZ'))
    trace = os.getenv('DETECT_RECREATE_TRACE_MODEL', 'True') == 'True'
    conf_thres = float(os.getenv('DETECT_CONF_THRES'))
    iou_thres = float(os.getenv('DETECT_IOU_THRES'))

    # Initialize
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA


    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, imgsz, tracedModelPath)

    if half:
        model.half()  # to FP16
    
    
    # Setup ImageZMQ Server
    imageHub = imagezmq.ImageHub()

    # Set Dataloader
    while True:

        # set True to speed up constant image size inference
        cudnn.benchmark = True  

        # Get names
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1


        # Iterate through all received frames
        while True:

            # print("Ready")

            # Get new image frame
            (videoSenderName, img_original) = imageHub.recv_image()

            # Start the timer
            start_time = time.time()




            # Convert color from BGR to RGB (OpenCV uses BGR by default)
            img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            # Resize and pad the image to maintain aspect ratio and fit the expected model input size
            # You need to define or import the 'letterbox' function used in your original code
            img_letterboxed = letterbox(img, new_shape=imgsz, stride=stride)[0]

            # Convert the letterboxed image to a PyTorch tensor
            img_tensor = torch.from_numpy(img_letterboxed).to(device)

            # Reorder dimensions: from HWC to CHW format and add a batch dimension
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            # Convert image data from uint8 to fp16/32 and normalize it to 0 - 1
            img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
            img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img_tensor)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)




            # Process detections
            jsonDetections = []
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    img1_shape = img_letterboxed.shape[:2]  # Height and Width of the processed image
                    img0_shape = img_original.shape[:2]  # Height and Width of the original image
                    det[:, :4] = scale_coords(img1_shape, det[:, :4], img0_shape).round()

                    
                    # Fill Detection JSON
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        detection = {
                            "class": names[int(cls)],
                            "confidence": float(conf),
                            "bbox": {
                                "x1": c1[0],
                                "y1": c1[1],
                                "x2": c2[0],
                                "y2": c2[1]
                            }
                        }
                        jsonDetections.append(detection)


            # Send Results
            imageHub.send_reply(json.dumps(jsonDetections, indent=4).encode())

            # Print Processing Time
            print(f"Proc. Time: {int((time.time() - start_time) * 1000)} ms")

            


detect()