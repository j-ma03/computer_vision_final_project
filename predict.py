import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO 
import os

# modelPath = '/Users/jaydenma/Documents/computer vision/computer_vision_final_project/best.pt'  # Replace with your model path
# modelPath = os.path.join(os.path.dirname(__file__), "best.pt")
# model = YOLO(modelPath)
# model = YOLO("yolov8n.pt")
# model = YOLO('best.pt')
model = YOLO('/Users/lucywu/computer_vision_final_project/best_1epoch.pt')


# frame = cv2.imread('/Users/jaydenma/Documents/computer vision/computer_vision_final_project/test.jpg')
image_path = os.path.join(os.path.dirname(__file__), "test.jpg")
frame = cv2.imread(image_path)

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
# cv2.imshow(frame)

cv2.imshow("Image", frame)  # "Image" is the window name, and 'frame' is the image to display
# cv2.waitKey(0)  # Waits for a key press to close the window
# cv2.destroyAllWindows()  # Closes the window