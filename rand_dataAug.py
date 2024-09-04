# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:14:37 2023

@author: jfarr
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import random


"Horizontal Flip"
def hor_flip(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load the image.")
        return

    # Horizontal flip using PyTorch's transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()
    ])
    flipped_img = transform(img)

    return img, flipped_img

"Vertical flip"
def ver_flip(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load the image.")
        return

    # Vertical flip using PyTorch's transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor()
    ])
    flipped_img = transform(img)

    return img, flipped_img

"Brightness"
def bright(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load the image.")
        return

    # Brightness adjustments using PyTorch
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4),
        transforms.ToTensor()
    ])
    flipped_img = transform(img)

    return img, flipped_img

"Rotation"
def rotate(img):
    angle = random.random() * 360
    if img is None:
        print("Failed to load the image.")
        return

    # rotation adjustments using PyTorch
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=angle),
        transforms.ToTensor()
    ])
    flipped_img = transform(img)

    return img, flipped_img

"Channel Shift"
def channel_shift(img):
   # Make a copy of the original image
    original_img = np.copy(img)

    # Apply channel shift to the copied image
    val = random.random() * 100
    value = int(random.uniform(-val, val))
    img = img + value

    # Clip pixel values to the valid range [0, 255]
    img[:,:,:][img[:,:,:]>255] = 255
    img[:,:,:][img[:,:,:]<0] = 0

    # Convert back to uint8
    img = img.astype(np.uint8)

    # Return both the original and altered images
    return original_img, img

"Vertical Shift"
def vertical_shift(image):
    # Make a copy of the original image
    original_image = np.copy(image)

    # Get image height and width
    height, width = image.shape[:2]
    
    #general shift range 
    shift_range = (-20,20);
    
    # Generate a random vertical shift value within the specified range
    shift_pixels = int(random.uniform(shift_range[0], shift_range[1]))

    # Define the transformation matrix for vertical shift
    shift_matrix = np.float32([[1, 0, 0], [0, 1, shift_pixels]])

    # Apply the vertical shift using warpAffine
    shifted_image = cv2.warpAffine(image, shift_matrix, (width, height))

    # Return both the original and the shifted images
    return original_image, shifted_image

"Horizontal Shift"
def horizontal_shift(image, ):
    # Make a copy of the original image
    original_image = np.copy(image)

    # Get image height and width
    height, width = image.shape[:2]
    
    #general shift range
    shift_range = (-20,20);
    
    # Generate a random horizontal shift value within the specified range
    shift_pixels = int(random.uniform(shift_range[0], shift_range[1]))

    # Define the transformation matrix for horizontal shift
    shift_matrix = np.float32([[1, 0, shift_pixels], [0, 1, 0]])

    # Apply the horizontal shift using warpAffine
    shifted_image = cv2.warpAffine(image, shift_matrix, (width, height))

    # Return both the original and the shifted images
    return original_image, shifted_image


"Random Function selection"
def rand_func_select(image):
    original_image = None  # Initialize variables
    altered_image = None

    available_functions = [hor_flip, ver_flip, bright, rotate, vertical_shift, horizontal_shift, channel_shift]

    # Randomly choose a function from the list
    chosen_function = random.choice(available_functions)

    # Apply the chosen function to the image
    if chosen_function == channel_shift:
        original_image, altered_image = chosen_function(cv2.imread(image_path))
        print("Applied Channel Shift")
    else:
        original_image, altered_image = chosen_function(cv2.imread(image_path))
        if chosen_function == vertical_shift:
            print("Applied Vertical Shift")
        elif chosen_function == horizontal_shift:
            print("Applied Horizontal Shift")
        elif chosen_function == hor_flip:
            print("Applied Horizontal Flip")
        elif chosen_function == ver_flip:
            print("Vertical Flip applied")
        elif chosen_function == rotate:
            print("Rotation applied")
        elif chosen_function == bright:
            print("Bright filter applied")
    
    return original_image, altered_image

# test code
image_path = 'Headshot.jpg'

original_image, altered_image = rand_func_select(image_path)

# Display the original and altered images
cv2.imshow('Original Image', original_image)
cv2.imshow('Altered Image', altered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
