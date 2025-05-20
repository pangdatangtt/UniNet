import os
import shutil
import random

images_dir = '../medical_segmentation/datasets/CVC-ColonDB/images'
masks_dir = '../medical_segmentation/datasets/CVC-ColonDB/masks'
train_image_dir = '../medical_segmentation/datasets/CVC-ColonDB/train/images'
train_gt_dir = '../medical_segmentation/datasets/CVC-ColonDB/train/masks'
test_image_dir = '../medical_segmentation/datasets/CVC-ColonDB/test/images'
test_gt_dir = '../medical_segmentation/datasets/CVC-ColonDB/test/masks'

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_gt_dir, exist_ok=True)

# os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_gt_dir, exist_ok=True)

image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)

assert len(image_files) == len(mask_files), "Images and masks must each contain 100 files."

random.seed(1) 
random.shuffle(image_files)

# 8:1:1
total_images = len(image_files)
train_size = int(total_images * 0.8)
val_size = int(total_images * 0.1)

train_images = image_files[:train_size]
val_images = image_files[train_size:train_size + val_size]
test_images = image_files[train_size + val_size:]

for img in train_images:
    shutil.copy(os.path.join(images_dir, img), os.path.join(train_image_dir, img))
    # mask_name = img.replace('.jpg', '_mask.png') 
    shutil.copy(os.path.join(masks_dir, img), os.path.join(train_gt_dir, img))

# for img in val_images:
#     shutil.copy(os.path.join(images_dir, img), os.path.join(val_dir, img))
#     mask_name = img.replace('.jpg', '_mask.png')
#     shutil.copy(os.path.join(masks_dir, mask_name), os.path.join(val_dir, mask_name))

for img in test_images:
    shutil.copy(os.path.join(images_dir, img), os.path.join(test_image_dir, img))
    # mask_name = img.replace('.jpg', '_mask.png')
    shutil.copy(os.path.join(masks_dir, img), os.path.join(test_gt_dir, img))

print("completing dataset splitting！")
