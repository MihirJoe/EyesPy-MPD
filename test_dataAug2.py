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
import PIL
from PIL import Image
import random

"Resize the image"
def resize_image(input_image_path, output_image_path, size):
    # Open the input image
    image = Image.open(input_image_path)

    # Ensure the image is a square
    width, height = image.size
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    new_image.paste(image, ((new_size - width) // 2, (new_size - height) // 2))

    # Resize the square image to the specified size
    resized_image = new_image.resize((size, size), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(output_image_path)

"Horizontal Flip"
def hor_flip(img):
    if img is None:
        print("Failed to load the image.")
        return None, None

    # Horizontal flip using OpenCV
    flipped_img = cv2.flip(img, 1)

    return img, flipped_img
"Vertical flip"
def ver_flip(img):
    if img is None:
        print("Failed to load the image.")
        return None, None

    # Vertical flip using OpenCV
    flipped_img = cv2.flip(img, 0)

    return img, flipped_img
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

    return img, adjusted_img

"Rotation"
def rotate(img):
    angle = random.random() * 360
    if img is None:
        print("Failed to load the image.")
        return

    # Get image height and width
    height, width = img.shape[:2]

    # Define the rotation center
    center = (width // 2, height // 2)

    # Generate a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

    return img, rotated_image

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
def rand_func_select(image_path):
    original_image = None
    altered_image = None

    available_functions = [hor_flip, ver_flip, bright, rotate, vertical_shift, horizontal_shift, channel_shift]

    img = cv2.imread(image_path)  # Load the image once

    chosen_function = random.choice(available_functions)

    if chosen_function == channel_shift:
        original_image, altered_image = chosen_function(np.copy(img))
        print("Applied Channel Shift")
    else:
        original_image, altered_image = chosen_function(np.copy(img))
        # Print statements for other functions
    # Check if the altered image is a valid NumPy array representing an image
   
    if isinstance(altered_image, np.ndarray):
        # Display the altered image using OpenCV
        print("The function :", chosen_function.__name__)
    else:
        # If altered_image is not valid, print the function that was attempted
        print("Failed to display the altered image.")
        print("The function that was attempted:", chosen_function.__name__)

    return original_image, altered_image


# test code
image_path = 'eye1.jpg'

# Resize the image to 256x256
resized_image_path = 'eye1_resize.jpg'
resize_image(image_path, resized_image_path, 256)

# Apply random transformation to the resized image
original_image, altered_image = rand_func_select(resized_image_path)

# Display the original and altered images using OpenCV
cv2.imshow('Original Image', original_image)
cv2.imshow('Altered Image', altered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
