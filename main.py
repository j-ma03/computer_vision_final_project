from predict import Predict
import os

def main():
    input_folder = 'input images'
    modelPath = './road features model/runs/detect/yolo11m_roadfeatures3/weights/best.pt'
    total_score = 0

    for i, filename in enumerate(os.listdir(input_folder)):
        if i >= 50:
            break
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            prediction = Predict()
            score = prediction.get_score(modelPath, image_path)
            print(f'Safety score for {filename}: {score}')
            total_score += score
    
    print(f'Total safety score for all images: {total_score}')


if __name__ == "__main__":
    main()