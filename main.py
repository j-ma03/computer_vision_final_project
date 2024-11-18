from predict import Predict
import os
import csv

def main():
    input_folder = 'input_images'
    modelPath = './road features model v3/runs/detect/yolo11m_roadfeatures2/weights/best.pt'
    scorePath = './scores v2.csv'
    path_scores = {}

    for subdir, _, files in os.walk(input_folder, topdown=True): # looping through the same images
        if subdir == input_folder:
            continue
        path_total_score = 0

        # print(f'Processing images in {subdir}')
        # csv_file = 'subdirs.csv'
        # with open(csv_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([subdir])
        print(subdir, '\n')
        for i, filename in enumerate(files):
            # if i >= 10:
            #     break
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                print(f'Processing {filename}')
                image_path = os.path.join(subdir, filename)
                # print(image_path)
                prediction = Predict()
                score = prediction.get_score(modelPath, image_path, scorePath)
                # print(f'Safety score for {filename}: {score}')
                path_total_score += score

                # with open(csv_file, mode='a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([filename, score])

        path_scores[subdir] = path_total_score

    print('Total scores for each path:')
    for path, score in path_scores.items():
        print(f'{path}: {score}')

    print('The safest path is:')
    safest_path = max(path_scores, key=path_scores.get)
    print(f'{safest_path}: {path_scores[safest_path]}')

    


if __name__ == "__main__":
    main()