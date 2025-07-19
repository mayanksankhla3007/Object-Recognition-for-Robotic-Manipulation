import os
import shutil
import random

# Paths
src_images = 'dataset_gripper_crops_labeled/images'
src_labels = 'dataset_gripper_crops_labeled/labels'
dst_root = 'gripper_crops_yolo'
train_ratio = 0.8

# Create YOLO directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(dst_root, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_root, split, 'labels'), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(src_images) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Split dataset
train_size = int(len(image_files) * train_ratio)
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# Function to copy files
def copy_files(file_list, split):
    for file in file_list:
        src_img = os.path.join(src_images, file)
        src_lbl = os.path.join(src_labels, file.replace('.jpg', '.txt').replace('.png', '.txt'))
        dst_img = os.path.join(dst_root, split, 'images', file)
        dst_lbl = os.path.join(dst_root, split, 'labels', os.path.basename(src_lbl))
        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# Copy files
copy_files(train_files, 'train')
copy_files(val_files, 'val')

print("✅ Dataset split into train/val folders!")
