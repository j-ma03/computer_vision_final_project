from predict import Predict
import os

def main():
    input_folder = 'input images'
    modelPath = './road features model/runs/detect/yolo11m_roadfeatures3/weights/best.pt'
    path_scores = {}

    for subdir, _, _ in os.walk(input_folder, topdown=True):
        if subdir == input_folder:
            continue
        path_total_score = 0
        print(f'Processing images in {subdir}')

        for i, filename in enumerate(os.listdir(input_folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                prediction = Predict()
                score = prediction.get_score(modelPath, image_path)
                print(f'Safety score for {filename}: {score}')
                path_total_score += score

        path_scores[subdir] = path_total_score

    print('Total scores for each path:')
    for path, score in path_scores.items():
        print(f'{path}: {score}')

    print('The safest path is:')
    safest_path = max(path_scores, key=path_scores.get)
    print(f'{safest_path}: {path_scores[safest_path]}')

    


if __name__ == "__main__":
    main()