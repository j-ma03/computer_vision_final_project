from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  

# Run batched inference on a list of images
results = model.predict("/Users/jaydenma/Documents/computer vision/computer_vision_final_project/test.jpg")  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
    print(boxes, masks, keypoints, probs, obb)  # print results