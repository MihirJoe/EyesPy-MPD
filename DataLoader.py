# -*- coding: utf-8 -*-
"""
Data Loader
"""

import pandas as pd
import torch
import os
import cv2
import numpy as np
import random


def resize_image(image, size):
    # Ensure the image is a square
    height, width, channels = image.shape
    new_size = max(width, height)

    # Create a white square canvas to paste our image on
    new_image = np.full((new_size, new_size, channels), 0, dtype=np.uint8)

    # Calculate the centering position
    center_x, center_y = (new_size - width) // 2, (new_size - height) // 2

    # Copy the original image onto the center of the canvas
    new_image[center_y:center_y+height, center_x:center_x+width] = image

    # Resize the square image to the specified size
    resized_image = cv2.resize(new_image, (size, size), interpolation=cv2.INTER_AREA)

    return resized_image
    
    

df = pd.read_excel('PracTraining.xlsx')

print(df)

images = []
targets = []
absolute_path = os.path.dirname(__file__)
relative_folder = "Images/DataLoad/InputFolder/"

#Might need to resize images here in loop
for i in range(len(df)):
    # Load and save the images
    file_name = df.loc[i,"file name"]
    
    relative_path = relative_folder + file_name
    
    imgpath = os.path.join(absolute_path, relative_path)
    
    img1  = cv2.imread(imgpath)
    img1 = resize_image(img1,200)
    
    images.append(img1)
    
    # Load and save the target values
    target = df.loc[i,"meaurement"]
    
    targets.append(target)
    
print(targets)


images_array = np.array(images)
targets_array = np.array(targets)
targets_array = targets_array.reshape(52, 1)

def create_train_mask(array):
    mask = np.random.choice([True, False], size=array.shape, p=[0.7, 0.3])
    return mask

def create_val_mask(mask):
    return np.logical_not(mask)

train_mask = create_train_mask(targets_array)
val_mask = create_val_mask(train_mask)

images_train_array = images_array[train_mask[:, 0]]
targets_train_array = targets_array[train_mask[:, 0]]

images_val_array = images_array[val_mask[:, 0]]
targets_val_array = targets_array[val_mask[:, 0]]


np.save('imgs_original_train.npy',images_train_array)
np.save('targets_original_train.npy',targets_train_array)

np.save('imgs_original_val.npy',images_val_array)
np.save('targets_original_val.npy',targets_val_array)

#img1array = images_array[0]




# USE THIS AFTER DATA AUGMENTATION##################################################################
# images_tensor = [torch.tensor(arr) for arr in images]
# images_tensor = [tensor.permute(2, 0, 1) for tensor in images_tensor]
# img1_tensor = images_tensor[0]
# targets_tensor = [torch.tensor(arr) for arr in targets]
####################################################################################################
# Next steps:
# 1. DONE Make excel file with the filenames and targets of 2 images from teams
# 2. DONE(DIFF SOLUTION)Use the CustomDataset class on that excel file -> dataset (size (2,2))
# 3. DONE(AFTER AUG)Save dataset as tensor file
# 4. Load tensor file into augmentation, produce datasetAugmented (size(20,2))
# 5. Run the datasetAugmented through the network





    