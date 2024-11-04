import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
import traceback

class CustomYOLO(YOLO):
    def __init__(self, model_path):
        super().__init__()
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            elif 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module):
                self.model = checkpoint['model']
            elif isinstance(checkpoint, dict):
                self.model = self.create_model()
                self.model.load_state_dict(checkpoint)
            else:
                raise TypeError(f"Expected state_dict to be dict-like, got {type(checkpoint)}.")
            print("Model loaded successfully.")
            self.model = self.model.float()  # Ensure all weights are in float32
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            exit(1)

    def create_model(self):
        # Define your model architecture here
        # This is a placeholder example, replace it with your actual model architecture
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, 1)
                self.conv2 = nn.Conv2d(16, 32, 3, 1)
                self.fc1 = nn.Linear(32 * 6 * 6, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.flatten(x, 1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return SimpleModel()

modelPath = '/Users/maver/Documents/Classes/Computer Vision/computer_vision_final_project/yolo11n.pt'

try:
    model = CustomYOLO(modelPath)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()
    exit(1)

frame = cv2.imread('/Users/maver/Documents/Classes/Computer Vision/computer_vision_final_project/cat.jpg')

# Predict using YOLO model
results = model.predict(frame)

# Draw bounding boxes and labels on the image
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the label
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()