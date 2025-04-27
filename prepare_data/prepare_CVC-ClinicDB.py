import os
import random
from shutil import copyfile
import pandas as  pd
import numpy as np
import cv2
import argparse

random.seed(1)

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='../medical_segmentation/datasets/CVC-ClinicDB/Original', type=str)
parser.add_argument('--save-folder', default='../medical_segmentation/datasets/CVC-ClinicDB', type=str)
config = parser.parse_args()

source_dir = config.data_folder
target_dir = config.save_folder

train_csv = np.array(pd.read_csv(os.path.join(target_dir, 'metadata.csv')))

all_path = [list() for i in range(612)]
train_path = []
train_gt_path = []
valid_path = []
valid_gt_path = []

for idx, line in enumerate(train_csv):
    suffix = line[-2].split('/')[-1]
    image_path = os.path.join(source_dir, str(suffix))
    gt_path = os.path.join(target_dir, 'Ground_Truth', str(suffix))
    all_path[idx].append(image_path)
    all_path[idx].append(gt_path)

random.shuffle(all_path)
# print(all_path)

# 8 : 1 : 1
for i in range(612):
    if i < 490:
        train_path.append(all_path[i][0])
        train_gt_path.append(all_path[i][-1])
    elif i >= 550:
        valid_path.append(all_path[i][0])
        valid_gt_path.append(all_path[i][-1])

if not os.path.exists(os.path.join(target_dir, 'train', 'images')):
    os.makedirs(os.path.join(target_dir, 'train', 'images'))
    os.makedirs(os.path.join(target_dir, 'train', 'masks'))
    os.makedirs(os.path.join(target_dir, 'test', 'images'))
    os.makedirs(os.path.join(target_dir, 'test', 'masks'))

target_train_path = os.path.join(target_dir, 'train', 'images')
for f in train_path:
    copyfile(f, os.path.join(target_train_path, os.path.basename(f)))

target_gt_path = os.path.join(target_dir, 'train', 'masks')
for f in train_gt_path:
    copyfile(f, os.path.join(target_gt_path, os.path.basename(f)))

target_test_path = os.path.join(target_dir, 'test', 'images')
for f in valid_path:
    copyfile(f, os.path.join(target_test_path, os.path.basename(f)))

target_test_gt_path = os.path.join(target_dir, 'test', 'masks')
for f in valid_gt_path:
    copyfile(f, os.path.join(target_test_gt_path, os.path.basename(f)))
