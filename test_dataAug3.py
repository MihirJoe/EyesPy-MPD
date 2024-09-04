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
import os
import csv
import pandas as pd

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
# def rand_func_select(image_path):
#     original_image = None
#     altered_image = None

#     available_functions = [hor_flip, ver_flip, bright, rotate, vertical_shift, horizontal_shift, channel_shift]

#     img = cv2.imread(image_path)  # Load the image once

#     chosen_function = random.choice(available_functions)

#     if chosen_function == channel_shift:
#         original_image, altered_image = chosen_function(np.copy(img))
#         print("Applied Channel Shift")
#     else:
#         original_image, altered_image = chosen_function(np.copy(img))
#         # Print statements for other functions
#     # Check if the altered image is a valid NumPy array representing an image
   
#     if isinstance(altered_image, np.ndarray):
#         # Display the altered image using OpenCV
#         print("The function :", chosen_function.__name__)
#     else:
#         # If altered_image is not valid, print the function that was attempted
#         print("Failed to display the altered image.")
#         print("The function that was attempted:", chosen_function.__name__)

#     return original_image, altered_image


# test code
img1 = 'eye1.jpg'
img2 = 'headshot.jpg'
img_array = [img1,img2]
# Resize the image to 256x256
counter = 1

# Specify the directory to save resized images
output_directory = "resized_images"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# New array to store resized image paths
resized_images_array = []

# ...

# Resize the image to 256x256
for x in range(len(img_array)):
    resized_image_path = os.path.join(output_directory, f'resized_image_{counter}.jpg')
    resize_image(img_array[x], resized_image_path, 256)
    counter += 1

    # Append the resized image path to the array
    resized_images_array.append(resized_image_path)

#Resized Image arrya
resized_images = []

# Load the resized images into the array
for resized_image_path in resized_images_array:
    resized_img = cv2.imread(resized_image_path)
    resized_images.append(resized_img)


altered_images_array = []

# Apply each transformation to the resized image
for x in range(len(resized_images)):
    count = 1
    
    # Horizontal flip
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = hor_flip(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Vertical Flip
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = ver_flip(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Brightness
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = bright(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Rotate
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = rotate(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Channel Shift
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = channel_shift(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Vertical Shift
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = vertical_shift(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)
    count += 1
    
    # Horizontal Shift
    altered_image_path = os.path.join(output_directory, f'resized_image_{x}_{count}.jpg')
    original_image, altered_image = horizontal_shift(resized_images[x])
    cv2.imwrite(altered_image_path, altered_image)
    altered_images_array.append(altered_image_path)

# Array to store altered images
altered_images = []

# Load the altered images into the array
for altered_image_path in altered_images_array:
    altered_img = cv2.imread(altered_image_path)
    altered_images.append(altered_img)

# Specify the CSV file path
csv_file_path = "image_data.csv"

# Specify the output directory
output_directory = "output_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Specify the CSV file path
csv_file_path = "image_data.csv"

# Create or open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csvfile:
    # Define the CSV writer
    csv_writer = csv.writer(csvfile)

# Create an empty list to store RGB data
rgb_data = []

# Iterate over the altered images
for idx, altered_image_path in enumerate(altered_images_array):
    # Read the altered image
    altered_img = cv2.imread(altered_image_path)
    height, width, _ = altered_img.shape
    
    # Iterate over pixels
    for i in range(height):
        for j in range(width):
            # Extract RGB values for each pixel
            pixel_row = i
            pixel_column = j
            red, green, blue = altered_img[i, j]
            
            # Append RGB data along with image path, pixel row, and pixel column
            rgb_data.append((altered_image_path, pixel_row, pixel_column, red, green, blue))

# Create a DataFrame directly with the correct columns
rgb_df = pd.DataFrame(rgb_data, columns=['Image_Path', 'Pixel_Row', 'Pixel_Column', 'Red', 'Green', 'Blue'])

# Set the MultiIndex with the correct names
rgb_df.set_index(['Image_Path', 'Pixel_Row', 'Pixel_Column'], inplace=True)

# Proceed to save the DataFrame to a CSV file
rgb_df.to_csv(csv_file_path)

# Save the DataFrame to a CSV file
rgb_df.to_csv(csv_file_path)

print("RGB values with MultiIndex saved to 'rgb_values_multiindex.csv' successfully.")