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
usingClasses = True

# Set source folder for images
source_folder = input("Enter full path (folder) where the images are stored: ")
# model_data_path = "/Users/mihirjoshi/Documents/OSU/2024-2025/Capstone/EyesPy/data/10-30-2024/model_data/subset/"

# Create destination folder to store images
destination_path = input("Enter full path (folder) where you want the Train/Test/Val Split Images to be stored: ")
# model_data_path = "/Users/mihirjoshi/Documents/OSU/2024-2025/Capstone/EyesPy/data/10-30-2024/model_data/subset/"

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
# The remaining 15% will go to the test folder

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

if testing: # print info about the number of images for each class
    print('Header (testing = True):')
    df.head()
    print('Open Classes:')
    df[df['class'] == 'Open'].count()
    print('Partial Classes:')
    df[df['class'] == 'Partial'].count()
    print('Closed Classes:')
    df[df['class'] == 'Closed'].count()

# Create a transform to convert the images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create the dataset for images
train_data = eyespy_nn.EyeDataset(data_path + train_label, df, transform=transform)
val_data = eyespy_nn.EyeDataset(data_path + val_label, df, transform=transform)

if testing: # print info about the number of images for each class
    print('Number of images in the training dataset:', len(train_data))
    print('Number of images in the validation dataset:', len(val_data))
    print(f'Shape of data: {train_data[0][1].shape}')


# only do these steps if the excel tracking is still using classes... 
if usingClasses:
    # Convert class names to numbers {open: 0,  partial: 1, closed: 2}
    unique_classes = df['class'].unique()
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

    df['class'] = df['class'].map(class_mapping)
    df.head()

    if testing: # print num and percent of each class in both the test and training data sets
        total_open = 0
        total_partial = 0
        total_closed = 0

        for data_sample in train_data:
            data_class = data_sample[2]

            if data_class == 0:
                total_open += 1

            elif data_class == 1:
                total_partial += 1

            else:
                total_closed += 1
        print(f"TRAINING Totals: \n  Open: {total_open}\n  Partial: {total_partial}\n  Closed: {total_closed}")
        print(f"TRAIN Percentages: \n  Open: {(total_open/len(train_data)) * 100:0.2f}%\n  Partial: {(total_partial / len(train_data)) * 100:0.2f}%\n  Closed: {(total_closed /len(train_data)) * 100:0.2f}%")

        total_open = 0
        total_partial = 0
        total_closed = 0

        for data_sample in val_data:
            data_class = data_sample[2]

            if data_class == 0:
                total_open += 1

            elif data_class == 1:
                total_partial += 1

            else:
                total_closed += 1
        print(f"VALIDATION Totals: \n  Open: {total_open}\n  Partial: {total_partial}\n  Closed: {total_closed}")
        print(f"VAL Percentages: \n  Open: {(total_open/len(val_data)) * 100:0.2f}%\n  Partial: {(total_partial / len(val_data)) * 100:0.2f}%\n  Closed: {(total_closed / len(val_data)) * 100:0.2f}%")

# Check if images were loaded correctly
dataset_correct = True

files = os.listdir(data_path + train_label)
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
image_files = [f for f in files if f.lower().endswith(image_extensions)]
index = 0

for img_idx in range(len(image_files)):
    index = img_idx
    curr_image = image_files[img_idx]

    filename_matches = (df[df['filename'] == curr_image]['filename'] == train_data[img_idx][0]).item()
    class_matches = (df[df['filename'] == curr_image]['class'] == train_data[img_idx][2]).item()
    vpf_matches = (df[df['filename'] == curr_image]['vpf'] == train_data[img_idx][3]).item()

    if not(filename_matches & class_matches & vpf_matches):
        dataset_correct = False
        break

if dataset_correct:
    print("Dataset created correctly!")
else:
    print(f"Uh oh :( the dataset was not correct at: {df.loc[index]}")


""" DEFINE MODEL AND HYPERPARAMETERS """
# Initialize the model, loss function, and optimizer
device = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# hyperparameters
num_epochs = 5
batch_size = 1
learning_rate = 0.001
criterion = nn.MSELoss()

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
        "loss": criterion,
        "dataset": labels_path, # assuming excel sheet is wellnamed... 
        "architecture": "Modified UNet"
    }
    )

# final prep
train_loader = eyespy_nn.create_loader(train_data, batch_size)
val_loader = eyespy_nn.create_loader(val_data, batch_size)

#run model training
eyespy_nn.training_and_validation(device, num_epochs, model, train_loader, val_loader, criterion, optimizer)

# Save the model weights
model_path = "model"
torch.save(model.state_dict(), model_path + '/Modified_Unet_{num_epochs}_epochs.pth') 