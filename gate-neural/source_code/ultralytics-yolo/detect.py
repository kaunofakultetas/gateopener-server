import os
import json
import cv2
import time
from ultralytics import YOLO
import imagezmq
import numpy as np
import torch




weights = os.getenv('DETECT_WEIGHTS', 'yolo11s.pt')
imgsz = int(os.getenv('DETECT_IMGSZ', "640"))
conf_thres = float(os.getenv('DETECT_CONF_THRES', "0.25"))
iou_thres = float(os.getenv('DETECT_IOU_THRES', "0.25"))



def compute_iou(box1, box2):
    # Calculate the (x, y) coordinates of the intersection of two boxes
    x1_inter = max(box1['x1'], box2['x1'])
    y1_inter = max(box1['y1'], box2['y1'])
    x2_inter = min(box1['x2'], box2['x2'])
    y2_inter = min(box1['y2'], box2['y2'])

    # Compute the area of intersection rectangle
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Compute the area of both the bounding boxes
    box1_area = (box1['x2'] - box1['x1'] + 1) * (box1['y2'] - box1['y1'] + 1)
    box2_area = (box2['x2'] - box2['x1'] + 1) * (box2['y2'] - box2['y1'] + 1)

    # Compute the Intersection over Union (IoU)
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def manual_nms(objects, iou_threshold=0.25):
    # Sort objects by confidence in descending order
    objects_sorted = sorted(objects, key=lambda x: x['confidence'], reverse=True)

    # List to hold final objects after NMS
    final_objects = []

    while len(objects_sorted) > 0:
        # Take the object with the highest confidence
        obj = objects_sorted.pop(0)
        final_objects.append(obj)

        # Compare IoU with remaining boxes and remove those with IoU greater than the threshold
        objects_sorted = [o for o in objects_sorted if compute_iou(obj['bbox'], o['bbox']) < iou_threshold]

    return final_objects




model = YOLO(weights)
# model()
# torch.set_num_threads(1)

imageHub = imagezmq.ImageHub()

while True:
    (videoSenderName, img_original) = imageHub.recv_image()

    start_time = time.time()

    predictions = model.predict(img_original, imgsz=imgsz, conf=conf_thres, iou=iou_thres, 
                                show=False, save=False, verbose=False, nms=False)
    torch.set_num_threads(1)

    jsonDetections = []
    
    for prediction in predictions:  # Assuming predictions is a list of prediction objects
        
        if prediction is None or len(prediction.boxes) == 0:
            continue
        
        # Assuming each prediction has a 'boxes' attribute that contains the detection data
        for box in prediction.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            conf = box.conf[0].cpu().item()  # Confidence
            cls = box.cls[0].cpu().item()  # Class ID

            bbox = {
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            }
            label = model.names[int(cls)]

            jsonDetections.append({
                "class": label,
                "confidence": float(conf),
                "bbox": bbox
            })

    # Apply Non-Maximum Suppression (NMS)
    jsonDetections = manual_nms(jsonDetections, iou_threshold=iou_thres)

    imageHub.send_reply(json.dumps(jsonDetections, indent=4).encode())

    print(f"Proc. Time: {int((time.time() - start_time) * 1000)} ms")
