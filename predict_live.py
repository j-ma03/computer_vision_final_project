import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO # Assuming you have a YOLO class in yolov5 module

modelPath = '/Users/jaydenma/Documents/mars rover/hopkins-rover/hammer_bottle detection/best_bw.pt'  # Replace with your model path
model = YOLO(modelPath)

# Initialize the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO model
    results = model.predict(frame)

    # Convert frame to RGB
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
            # cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            # cv2.putText(img, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Live Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()