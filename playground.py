from predict import Predict
import os
import cv2
from ultralytics import YOLO
import traceback
import csv
import sys

input_folder = 'input_images'
modelPath = './road features model v2/weights/best.pt'
scorePath = './scores v2.csv'
imgPath = 'input_images/path3/5_035_jpg.rf.451b3e249b925c61662fe3f2e3a19e28.jpg'
scores = {}

prediction = Predict()
prediction.get_score(modelPath, imgPath, scorePath)

with open(scorePath, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        scores[row['class']] = float(row['score'])

try:
    model = YOLO(modelPath)
except KeyError as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    traceback.print_exc()
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()
    exit(1)

frame = cv2.imread(imgPath) # input image

# Predict using YOLO model
results = model.predict(frame)

# Process results list
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        formatted_result = {
            'class': class_name,
            'confidence': confidence.item(),
            'bbox': [x1, y1, x2, y2]
        }
        if (confidence >= 0.5):
            # cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            # cv2.putText(img, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # print(class_name, scores[class_name], confidence)

    # Display the frame
    cv2.imwrite("output.jpg", frame)