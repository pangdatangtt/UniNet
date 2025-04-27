import os
import random
from shutil import copyfile
import pandas as  pd
import numpy as np
import cv2
import argparse

random.seed(1)

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='/data/disk2T1/guoj/ISIC2018/original', type=str)
parser.add_argument('--save-folder', default='/data/disk2T1/guoj/ISIC2018', type=str)
config = parser.parse_args()

source_dir = config.data_folder
target_dir = config.save_folder

train_csv = np.array(pd.read_csv(
    os.path.join(source_dir, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')))
valid_csv = np.array(pd.read_csv(
    os.path.join(source_dir, 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')))

train_normal_path = []
valid_normal_path = []
valid_abnormal_path = []

for line in train_csv:
    if line[2] == 1:
        train_normal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Training_Input', line[0] + '.jpg'))

for line in valid_csv:
    if line[2] == 1:
        valid_normal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Validation_Input', line[0] + '.jpg'))
    else:
        valid_abnormal_path.append(os.path.join(source_dir, 'ISIC2018_Task3_Validation_Input', line[0] + '.jpg'))

target_train_normal_dir = os.path.join(target_dir, 'train', 'NORMAL')
if not os.path.exists(target_train_normal_dir):
    os.makedirs(target_train_normal_dir)

target_test_normal_dir = os.path.join(target_dir, 'test', 'NORMAL')
if not os.path.exists(target_test_normal_dir):
    os.makedirs(target_test_normal_dir)

target_test_abnormal_dir = os.path.join(target_dir, 'test', 'ABNORMAL')
if not os.path.exists(target_test_abnormal_dir):
    os.makedirs(target_test_abnormal_dir)

for f in train_normal_path:
    copyfile(f, os.path.join(target_train_normal_dir, os.path.basename(f)))

for f in valid_normal_path:
    copyfile(f, os.path.join(target_test_normal_dir, os.path.basename(f)))

for f in valid_abnormal_path:
    copyfile(f, os.path.join(target_test_abnormal_dir, os.path.basename(f)))
