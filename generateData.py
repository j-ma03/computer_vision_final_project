# https://www.kaggle.com/code/lucywu012/get-small-subset-of-original-dataset/edit

# method used to generate 100 random images for annotation purposes
# technically does not work as is because the data is processed through kaggle, but can be helpful

import os
import random
from shutil import copy, make_archive, rmtree

random.seed()

data_root = '/kaggle/input/100k-vehicle-dashcam-image-dataset'
k = 100  # Randomly select 100 images from the train data folder

# Remove the dataset folder if it exists to start fresh
if os.path.exists('./dataset'):
    rmtree('./dataset')  # Deletes the entire dataset folder and its contents

os.makedirs('./dataset', exist_ok=True)  # Create a fresh dataset folder

for d in ['train']:
    # List all images in the train folder
    dir_path = os.path.join(data_root, d)
    files = os.listdir(dir_path)
    
    # Copy images to the target folder
    target_dir = os.path.join('dataset', d)
    os.makedirs(target_dir, exist_ok=True)
    for f in random.choices(files, k=k):  # Randomly select k images and copy them to the target folder
        src_file = os.path.join(dir_path, f)
        copy(src_file, target_dir)

# Zip the generated files
make_archive(base_name='road_dataset5', format='zip', root_dir='dataset')