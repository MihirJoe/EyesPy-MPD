# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:22:37 2023

@author: allen
"""

"""
The purpose of this neural network is to input images of eyes and output the
measurement of vertical palpebral fissure
"""
#%%packages
import torch
import torch.nn as nn
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

#Makes the cpu act as a gpu in case of not enough ram
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Model

"""
Specify the shape of network architecture
"""
class ModelWithConv(nn.Module):
    def __init__(self):
        super(ModelWithConv, self).__init__()
       
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Fully connected layers (128 is number of neurons in fully connected layers currently 2 layers)
        """Replace image width and height with whatever ours is
        double // divides then uses floor """
        self.fc1 = nn.Linear(32 * (image_width // 4) * (image_height // 4), 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(128, 1)
   
    def forward(self, img):
        img = self.conv1(img)
        img = self.relu1(img)
        img = self.maxpool1(img)
      
        img = self.conv2(img)
        img = self.relu2(img)
        img = self.maxpool2(img)
       
        img = img.view(img.size(0), -1)  # Flatten the output for the fully connected layers to 1-D vector
       
        img = self.fc1(img)
        img = self.relu3(img)
        img = self.fc2(img)
        img = self.relu4(img)
        output = self.output_layer(img)
        return output
   
    def loss(self, img, palp_target):

    # palp is equal to the img once passes through the network
        palp = self.net(img)

    #automatically dimensions as a column vector equal to input

        N = len(palp)  # setting N equal to the however many elements are in palp
        loss = 1/N*torch.mean((palp - palp_target)**2)  # mean squared error

        return loss
 

#%%

model = ModelWithConv().to(device)
#________________________________________

"""
lr=0.001 is pretty good / lr = learning rate - via back propogation how network minimizes loss using partial derivatives
 - takes derivative of loss with respect to all weights and biases and puts them into a large matrix and minimizes it
"""
# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001) 
""" The value of the learning rate controbutes to the
 gradient of the partial derivatives, increasing can make the optimizer unstable - learning rate scheduling to
 increase or decresase the learning rate as time goes on to start accurate and build to run quicker
"""
epochs = 30000 # epoch is one round of training"""

training_losses = []
validation_losses = []
epochsarray = []

# Training
def train(epoch):
    model.train()

    def closure():
        optimizer.zero_grad()
        training_loss = model.loss(t_img_shuffled, t_palp_target_shuffled) #"""calculates training loss using the training shuffled images"""

        training_loss.backward()
        return training_loss

    training_loss = optimizer.step(closure) #one step using Adam optimizer
    
    
    model.eval() #set model equal to eval mode to avoid affecting the way the network trains 
    #shows independence from set of data youre using the train with
    validation_loss = model.loss(v_img_shuffled, v_palp_target_shuffled)
    
    
    training_losses.append(training_loss.item())
    validation_losses.append(validation_loss.item())
    epochsarray.append(epoch)
    
    
    print(f"Epoch [{epoch}]")
    print(f"Training Loss: {training_loss.item()}")
    print(f"Validation Loss: {validation_loss.item()}")

#%%
"""load data """


#%%
""" training """
print('start training...')
tic = time.time() #takes current value of time and sets it equal to tic
for epoch in range(1, epochs + 1):
    # Shuffle the data to make sure it does not overfit based on the listed data/ ensures generalizability
    permutation1 = torch.randperm(t_img_tensor.size(0))
    t_img_shuffled = t_img_tens[permutation1]
    t_palp_target_shuffled = t_palp_target_tens[permutation1]
   #may have to normalize the image data
    #for both the traning and validation data
    permutation2 = torch.randperm(v_img_tensor.size(0))
    v_img_shuffled = v_img_tens[permutation2]
    v_palp_target_shuffled = v_palp_target_tens[permutation2]

    
    train(epoch) #using defined training function
    
toc = time.time()
print(f'total training time: {toc - tic}')

#%% Saving Loss data
training_losses = np.array(training_losses) #changes tensor to numpy array so you can save as a txt
validation_losses = np.array(validation_losses)
epochsarray = np.array(epochsarray)

folder_path = 'C:/Users/allen/Documents/MatlabML/Palpebral Fissure'

def saveNPasTxt(folder_path, file_name, var):
    file_path = folder_path + file_name
    np.savetxt(file_path, var, fmt='%.15f', delimiter='\t')

saveNPasTxt(folder_path, 'palp_fissure_val_losses.txt', validation_losses)
saveNPasTxt(folder_path, 'palp_fissure_train_losses.txt', training_losses)
saveNPasTxt(folder_path, 'palp_fissure_epochs.txt', epochsarray)

"""Next steps: Figure out how to load a bunch of images into python and 
need to determine whether the the thing we are using as our input is the whole image"""
