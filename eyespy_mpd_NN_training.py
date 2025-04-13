import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim
import tifffile 
import random
import shutil

WANDB_FLAG = False
if WANDB_FLAG:
    import wandb

import eyespy_mpd_NN as eyespy_nn

testing = True 

# Set source folder for images
source_folder = input("Enter full path (folder) where the images are stored: ")

# Create destination folder to store images
destination_path = input("Enter full path (folder) where you want the Train/Test/Val Split Images to be stored: ")

# Create the Train, Validation, and Test folders
folders = ["Train", "Validation", "Test"]
for folder in folders:
    os.makedirs(os.path.join(destination_path, folder), exist_ok=True)

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Shuffle the list of image files randomly
random.shuffle(image_files)

# Calculate the number of images for each folder
total_images = len(image_files)
train_split = int(total_images * 0.7)  # 70% for training
val_split = int(total_images * 0.2)   # 20% for validation
# The remaining 10% will go to the test folder

# Distribute the images into the folders
for i, image in enumerate(image_files):
    source = os.path.join(source_folder, image)
    if i < train_split:
        destination = os.path.join(destination_path, "Train", image)
    elif i < train_split + val_split:
        destination = os.path.join(destination_path, "Validation", image)
    else:
        destination = os.path.join(destination_path, "Test", image)
    
    shutil.copy2(source, destination)

print(f"Total images: {total_images}")
print(f"Images in Train: {train_split}")
print(f"Images in Validation: {val_split}")
print(f"Images in Test: {total_images - train_split - val_split}")


# Update these parameters
data_path = destination_path
train_label =  'Train/'
val_label = 'Validation/'

# get path to the excel file
labels_path = input("Enter full path (excel file) where the filenames, labels, and measures are stored: ")
df = pd.read_excel(labels_path)

# Make sure we have all required columns
required_columns = ['filename', 'vpf', 'mrd1', 'vpf_expected', 'mrd1_expected']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in Excel file: {missing_columns}")

if testing: # print info about the data
    print('Header (testing = True):')
    print(df.head())
    print(f"Total number of data points: {len(df)}")
    print("\nStatistics for input features:")
    print(f"VPF mean: {df['vpf'].mean():.2f}, std: {df['vpf'].std():.2f}")
    print(f"MRD1 mean: {df['mrd1'].mean():.2f}, std: {df['mrd1'].std():.2f}")
    print("\nStatistics for target values:")
    print(f"VPF_expected mean: {df['vpf_expected'].mean():.2f}, std: {df['vpf_expected'].std():.2f}")
    print(f"MRD1_expected mean: {df['mrd1_expected'].mean():.2f}, std: {df['mrd1_expected'].std():.2f}")

# Create a transform to convert the images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create the dataset for images
train_data = eyespy_nn.EyeDataset(data_path + train_label, df, transform=transform)
val_data = eyespy_nn.EyeDataset(data_path + val_label, df, transform=transform)

if testing: # print info about datasets
    print('Number of images in the training dataset:', len(train_data))
    print('Number of images in the validation dataset:', len(val_data))
    if len(train_data) > 0:
        print(f'Shape of data: {train_data[0][1].shape}')
        print(f'Sample data: filename={train_data[0][0]}, vpf={train_data[0][2]}, mrd1={train_data[0][3]}, vpf_expected={train_data[0][4]}, mrd1_expected={train_data[0][5]}')

# Check if images were loaded correctly
dataset_correct = True

files = os.listdir(data_path + train_label)
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
image_files = [f for f in files if f.lower().endswith(image_extensions)]
index = 0

for img_idx in range(len(image_files)):
    index = img_idx
    curr_image = image_files[img_idx]

    try:
        # Check if filename matches
        filename_matches = (df[df['filename'] == curr_image]['filename'] == train_data[img_idx][0]).item()
        
        # Check if other values match
        vpf_matches = (df[df['filename'] == curr_image]['vpf'] == train_data[img_idx][2]).item()
        mrd1_matches = (df[df['filename'] == curr_image]['mrd1'] == train_data[img_idx][3]).item()
        vpf_expected_matches = (df[df['filename'] == curr_image]['vpf_expected'] == train_data[img_idx][4]).item()
        mrd1_expected_matches = (df[df['filename'] == curr_image]['mrd1_expected'] == train_data[img_idx][5]).item()

        if not(filename_matches & vpf_matches & mrd1_matches & vpf_expected_matches & mrd1_expected_matches):
            dataset_correct = False
            break
    except (IndexError, KeyError) as e:
        print(f"Error checking dataset at index {img_idx}, file {curr_image}: {e}")
        dataset_correct = False
        break

if dataset_correct:
    print("Dataset created correctly!")
else:
    print(f"Uh oh :( the dataset was not correct at index: {index}, filename: {curr_image}")
    print("Please verify that your Excel file has the correct columns and matches the image files.")


""" DEFINE MODEL AND HYPERPARAMETERS """
# Initialize the model, loss function, and optimizer
device = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# hyperparameters
num_epochs = 5
batch_size = 1
learning_rate = 0.001
criterion = nn.MSELoss()  # MSE loss is appropriate for regression tasks

# initialize model
model = eyespy_nn.ModifiedUNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if WANDB_FLAG:
    wandb.init(
    # set wandb project
    project="eyespy-mpd",

    # track hyperparameters
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "loss": "MSE",
        "dataset": labels_path, # assuming excel sheet is wellnamed... 
        "architecture": "Modified UNet with dual output"
    }
    )

# final prep
train_loader = eyespy_nn.create_loader(train_data, batch_size)
val_loader = eyespy_nn.create_loader(val_data, batch_size)

#run model training
eyespy_nn.training_and_validation(device, num_epochs, model, train_loader, val_loader, criterion, optimizer)

# Save the model weights
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, f'Modified_Unet_{num_epochs}_epochs.pth')) 