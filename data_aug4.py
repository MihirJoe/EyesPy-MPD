# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:20:18 2024

@author: campb
"""

import pandas as pd
import torch
import os
import cv2
import numpy as np
import random





"Horizontal Flip"
def hor_flip(img):
    if img is None:
        print("Failed to load the image.")
        return None, None

    # Horizontal flip using OpenCV
    flipped_img = cv2.flip(img, 1)
    

    return flipped_img
"Vertical flip"
def ver_flip(img):
    if img is None:
        print("Failed to load the image.")
        return None, None

    # Vertical flip using OpenCV
    flipped_img = cv2.flip(img, 0)

    return flipped_img
"Brightness"
def bright(img):
    if img is None:
        print("Failed to load the image.")
        return None, None

    # Convert BGR image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a random brightness value between 0.1 and 0.6
    brightness = random.uniform(0.1, 0.6)

    # Scale the V channel by the brightness factor
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * brightness, 0, 255).astype(np.uint8)

    # Convert the HSV image back to BGR
    adjusted_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_img

"Rotation"
def rotate(img):
    angle = random.random() * 360
    if img is None:
        print("Failed to load the image.")
        return

    # Get image height and width
    height, width, channels = img.shape

    # Define the rotation center
    center = (width // 2, height // 2)

    # Generate a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

    return rotated_image

"Channel Shift"
def channel_shift(img):
    # Create a copy of the image to avoid changing the original
    img_shifted = np.array(img, copy=True)
    
    # Apply channel shift to the copied image for each channel separately
    for i in range(3):  # Assuming img has three channels (RGB)
        val = random.randrange(70,99) # Random value to shift
        value = int(random.uniform(-val, val))
        img_shifted[:, :, i] = img_shifted[:, :, i] + value

    # Ensure the values are in the valid range [0, 255] and convert to uint8
    img_shifted = np.clip(img_shifted, 0, 255).astype(np.uint8)

    # Return the shifted image
    return img_shifted

"Vertical Shift"
def vertical_shift(image):
    

    # Get image height and width
    height, width, channels = image.shape
    
    #general shift range 
    shift_range = (-20,20);
    
    # Generate a random vertical shift value within the specified range
    shift_pixels = int(random.uniform(shift_range[0], shift_range[1]))

    # Define the transformation matrix for vertical shift
    shift_matrix = np.float32([[1, 0, 0], [0, 1, shift_pixels]])

    # Apply the vertical shift using warpAffine
    shifted_image = cv2.warpAffine(image, shift_matrix, (width, height))

    # Return both the original and the shifted images
    return shifted_image

"Horizontal Shift"
def horizontal_shift(image, ):
    
    # Get image height and width
    height, width, channels = image.shape
    
    #general shift range
    shift_range = (-20,20);
    
    # Generate a random horizontal shift value within the specified range
    shift_pixels = int(random.uniform(shift_range[0], shift_range[1]))

    # Define the transformation matrix for horizontal shift
    shift_matrix = np.float32([[1, 0, shift_pixels], [0, 1, 0]])

    # Apply the horizontal shift using warpAffine
    shifted_image = cv2.warpAffine(image, shift_matrix, (width, height))

    # Return both the original and the shifted images
    return shifted_image




images_train = np.load('imgs_original_train.npy')
targets_train = np.load('targets_original_train.npy')

altered_targets_list_train = []
altered_images_list_train = []

# Apply each transformation to the resized image
for x in range(len(images_train)):
   
    # Horizontal flip
    altered_image = hor_flip(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])

    
    # Vertical Flip
    altered_image = ver_flip(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])

    
    # Brightness
    altered_image = bright(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])
    
    # Rotate
    altered_image = rotate(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])
    
    # Channel Shift
    altered_image = channel_shift(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])
    
    # Vertical Shift
    altered_image = vertical_shift(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])
    
    # Horizontal Shift
    altered_image = horizontal_shift(images_train[x])
    altered_images_list_train.append(altered_image)
    altered_targets_list_train.append(targets_train[x])
    
    
    
altered_targets_array_train = np.array(altered_targets_list_train)
altered_images_array_train = np.array(altered_images_list_train)
   
    
full_image_array_train = np.concatenate((images_train, altered_images_array_train), axis=0)
full_target_array_train = np.concatenate((targets_train, altered_targets_array_train), axis=0)
#full_target_array_train = full_target_array_train.reshape(144, 1)

np.save('full_image_train.npy',full_image_array_train)
np.save('full_target_train.npy',full_target_array_train)



images_val = np.load('imgs_original_val.npy')
targets_val = np.load('targets_original_val.npy')

altered_targets_list_val = []
altered_images_list_val = []

# Apply each transformation to the resized image
for x in range(len(images_val)):
   
    # Horizontal flip
    altered_image = hor_flip(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])

    
    # Vertical Flip
    altered_image = ver_flip(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])

    
    # Brightness
    altered_image = bright(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])
    
    # Rotate
    altered_image = rotate(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])
    
    # Channel Shift
    altered_image = channel_shift(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])
    
    # Vertical Shift
    altered_image = vertical_shift(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])
    
    # Horizontal Shift
    altered_image = horizontal_shift(images_val[x])
    altered_images_list_val.append(altered_image)
    altered_targets_list_val.append(targets_val[x])
    
    
    
altered_targets_array_val = np.array(altered_targets_list_val)
altered_images_array_val = np.array(altered_images_list_val)
   
    
full_image_array_val = np.concatenate((images_val, altered_images_array_val), axis=0)
full_target_array_val = np.concatenate((targets_val, altered_targets_array_val), axis=0)
#full_target_array_val = full_target_array_val.reshape(, 1)

np.save('full_image_val.npy',full_image_array_val)
np.save('full_target_val.npy',full_target_array_val)

# USE THIS AFTER DATA AUGMENTATION##################################################################


#Display the altered image
# cv2.imshow('Altered Image', full_image_array_val[6])

# #Wait for a key event and close the window
# key = cv2.waitKey(0)

# #Check if the pressed key is the 'esc' key (27 is the ASCII code for the 'esc' key)
# if key == 27:
#     cv2.destroyAllWindows()



