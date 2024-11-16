import cv2
from ultralytics import YOLO
import traceback
import csv

class Predict():
#     def __init__(self):
#         pass

#     def get_score(self, img):
#         modelPath = '/Users/jaydenma/Documents/computer vision/computer_vision_final_project/road features model/runs/detect/yolo11m_roadfeatures3/weights/last.pt'

    def get_score(self, modelPath, imgPath): # get the safety score of one image
        scorePath = './scores.csv'
        totalScore = 0

        # Read scores from CSV file into a dictionary
        scores = {}
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
                # cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                # cv2.putText(img, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f'Class: {class_name}, Conf: {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # print(class_name, scores[class_name], confidence)
                print(class_name, scores[class_name], confidence)
                totalScore += scores[class_name] 

        # Display the frame
        # cv2.imwrite("output.jpg", frame)
        return totalScore