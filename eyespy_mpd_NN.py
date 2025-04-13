"""Import necessary Libraries"""

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

WANDB_FLAG = False
if WANDB_FLAG:
    import wandb

"""Nueral Network Class"""
class ModifiedUNet(nn.Module):
    def __init__(self):
        super(ModifiedUNet, self).__init__()

        # Encoder (Contracting Path)
        self.inc = self.double_conv(1, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024)

        # Decoder (Expanding Path)
        self.up1 = self.up(1024, 512)
        self.up2 = self.up(512, 256)
        self.up3 = self.up(256, 128)
        self.up4 = self.up(128, 64)

        # Final convolution
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for final output
        self.fc1 = nn.Linear(3, 128)  # Changed from 4 to 3 to remove class_label input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output two values: vpf_predicted and mrd1_predicted

    def forward(self, x, vpf, mrd1):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv(x.size(1), 512)(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv(x.size(1), 256)(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv(x.size(1), 128)(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv(x.size(1), 64)(x)

        x = self.outc(x)

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Unsqueeze inputs to make them 2D if needed
        vpf = vpf.unsqueeze(1).float()
        mrd1 = mrd1.unsqueeze(1).float()

        # Concatenate with vpf and mrd1
        x = torch.cat([x, vpf, mrd1], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  # Returns a tensor with 2 values: [vpf_predicted, mrd1_predicted]

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def up_conv(self, in_channels, out_channels):
        return self.double_conv(in_channels, out_channels)


"""Functions for Dataset Management"""
class EyeDataset(Dataset):
    def __init__(self, directory, dataframe, transform=None):
        self.directory = directory
        self.dataframe = dataframe
        self.transform = transform
        self.filenames = self.__get_valid_filenames()

    def __get_valid_filenames(self):
        # Get all image files in the directory
        all_files = [f for f in os.listdir(self.directory) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        
        # Filter files that exist in both the directory and the dataframe
        valid_files = [f for f in all_files if f in self.dataframe['filename'].values]
        return valid_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.directory, filename)

        # get corresponding row from dataframe
        row = self.dataframe[self.dataframe['filename'] == filename].iloc[0]

        # Load image
        image = self.__load_image(img_path)

        # Get values from dataframe
        vpf = row['vpf']
        mrd1 = row['mrd1']
        vpf_expected = row['vpf_expected']
        mrd1_expected = row['mrd1_expected']

        return filename, image, vpf, mrd1, vpf_expected, mrd1_expected
    
    def __load_image(self, img_path):
        image = Image.open(img_path)
        if self.transform: # Dynamically apply data transformation
            image = self.transform(image)
        return image
    
    def save(self, model_path="./Modified_UNet_WandB.pth"):
        torch.save(self.state_dict(), model_path)
    
# Training and validation loops
def train_epoch(device, model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    count = 0
    for _, img, vpf, mrd1, vpf_expected, mrd1_expected in loader:
        img = img.to(device=device, dtype=torch.float32)
        
        if torch.backends.mps.is_available():
            # Pytorch only converts MPS tensors to float32
            vpf = vpf.to(device=device, dtype=torch.float32)
            mrd1 = mrd1.to(device=device, dtype=torch.float32)
            vpf_expected = vpf_expected.to(device=device, dtype=torch.float32)
            mrd1_expected = mrd1_expected.to(device=device, dtype=torch.float32)
        else:
            vpf = vpf.to(device).float()
            mrd1 = mrd1.to(device).float()
            vpf_expected = vpf_expected.to(device).float()
            mrd1_expected = mrd1_expected.to(device).float()

        optimizer.zero_grad()
        
        output = model(img, vpf, mrd1)  # Model outputs [vpf_predicted, mrd1_predicted]
        
        # Create target tensor with both expected values
        targets = torch.stack([vpf_expected, mrd1_expected], dim=1)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print(f"batch {count+1}")
        count += 1

    curr_loss = running_loss / len(loader)
    if WANDB_FLAG:
        wandb.log({"train_MSE": curr_loss})
    return curr_loss

def create_loader(train_dataset, batch_size):
    torch.manual_seed(0)  # For reproducibility of random numbers in PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Creates a training DataLoader from this Dataset

    return train_loader

""" Functions for training """
def validate(device, model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for _, img, vpf, mrd1, vpf_expected, mrd1_expected in loader:
            img = img.to(device=device, dtype=torch.float32)
            
            if torch.backends.mps.is_available():
                # Pytorch only converts MPS tensors to float32
                vpf = vpf.to(device=device, dtype=torch.float32)
                mrd1 = mrd1.to(device=device, dtype=torch.float32)
                vpf_expected = vpf_expected.to(device=device, dtype=torch.float32)
                mrd1_expected = mrd1_expected.to(device=device, dtype=torch.float32)
            else:
                vpf = vpf.to(device).float()
                mrd1 = mrd1.to(device).float()
                vpf_expected = vpf_expected.to(device).float()
                mrd1_expected = mrd1_expected.to(device).float()
            
            output = model(img, vpf, mrd1)
            
            # Create target tensor
            targets = torch.stack([vpf_expected, mrd1_expected], dim=1)
            
            loss = criterion(output, targets)
            running_loss += loss.item()
        
        curr_loss = running_loss / len(loader)
        if WANDB_FLAG:
            wandb.log({"val_MSE": curr_loss})
    return curr_loss

def training_and_validation(device, num_epochs, model, train_loader, val_loader, criterion, optimizer):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(device, model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        val_loss = validate(device, model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    # Plotting the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()
