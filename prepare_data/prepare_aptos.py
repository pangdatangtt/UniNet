import os
import random
from shutil import copyfile
import pandas as pd
import numpy as np
from skimage.measure import regionprops, label
from operator import attrgetter
import cv2
import argparse


def fill_crop(img, min_idx, max_idx):
    crop = np.zeros(np.array(max_idx, dtype='int16') - np.array(min_idx, dtype='int16'), dtype=img.dtype)
    img_shape, start, crop_shape = np.array(img.shape), np.array(min_idx, dtype='int16'), np.array(crop.shape),
    end = start + crop_shape
    # Calculate crop slice positions
    crop_low = np.clip(0 - start, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = (slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    pos = np.clip(start, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = (slice(low, high) for low, high in zip(pos, end))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]
    return crop


def fundus_crop(image, shape=[512, 512], margin=5):
    mask = (image.sum(axis=-1) > 30)
    mask = label(mask)
    regions = regionprops(mask)
    region = max(regions, key=attrgetter('area'))

    len = (np.array(region.bbox[2:4]) - np.array(region.bbox[0:2])).max()
    bbox = np.concatenate([np.array(region.centroid) - len / 2, np.array(region.centroid) + len / 2]).astype('int16')

    image_b = fill_crop(image, [bbox[0] - margin, bbox[1] - margin, 0], [bbox[2] + margin, bbox[3] + margin, 3])
    image_b = cv2.resize(image_b, shape, interpolation=cv2.INTER_LINEAR)
    return image_b


parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='/data/disk2T1/jialz/APTOS/original', type=str)
parser.add_argument('--save-folder', default='/data/disk2T1/guoj/APTOS', type=str)
config = parser.parse_args()

random.seed(1)

source_dir = config.data_folder
target_dir = config.save_folder

data_csv = np.array(pd.read_csv(os.path.join(source_dir, 'train.csv')))
normal_path = []
abnormal_path = []

for line in data_csv:
    if line[1] == 0:
        normal_path.append(os.path.join(source_dir, 'train_images', line[0] + '.png'))
    else:
        abnormal_path.append(os.path.join(source_dir, 'train_images', line[0] + '.png'))

random.shuffle(normal_path)
train_normal_path = normal_path[:1000]
test_normal_path = normal_path[1000:]

if not os.path.exists(os.path.join(target_dir, 'train', 'NORMAL')):
    os.makedirs(os.path.join(target_dir, 'train', 'NORMAL'))
    os.makedirs(os.path.join(target_dir, 'test', 'NORMAL'))
    os.makedirs(os.path.join(target_dir, 'test', 'ABNORMAL'))

for path in train_normal_path:
    image = cv2.imread(path)
    image = fundus_crop(image, shape=[512, 512], margin=5)
    cv2.imwrite(os.path.join(target_dir, 'train', 'NORMAL', os.path.basename(path)), image)

for path in test_normal_path:
    image = cv2.imread(path)
    image = fundus_crop(image, shape=[512, 512], margin=5)
    cv2.imwrite(os.path.join(target_dir, 'test', 'NORMAL', os.path.basename(path)), image)

for path in abnormal_path:
    image = cv2.imread(path)
    image = fundus_crop(image, shape=[512, 512], margin=5)
    cv2.imwrite(os.path.join(target_dir, 'test', 'ABNORMAL', os.path.basename(path)), image)
