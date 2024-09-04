# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:27:10 2023

@author: allen
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
# ######Latex stuff############################################################################
# legend_settings = {
#     "legend.edgecolor": 'black',
#     "legend.facecolor": 'white',
#     "legend.framealpha": 0.8,
#     "legend.borderaxespad": 2.0,  # Adjust this value to control border width indirectly
#     "legend.shadow": False,
#     "legend.fancybox": False
# }

# # Update rcParams with custom legend settings
# plt.rcParams.update(legend_settings)

# plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"

# params = {"text.usetex" : True,
#           # "font.size" : 12,
#           "font.family" : "lmodern",
#           "legend.fontsize": "large",
#           "figure.figsize": (15, 5),
#           "axes.labelsize": "xx-large",
#           "axes.titlesize":"xx-large",
#           "xtick.labelsize":"large",
#           "ytick.labelsize":"large"}
# plt.rcParams.update(params)
# plt.rc("text.latex", preamble=r"\usepackage{underscore}") # This line is to ensure that LaTex read "_" as "\_" in filenames
# #################################################################################################################################

#
  #
    #
    #
      #
        #
        #
          #
            #
            #
              #
                #
########################################################################################
####Run this in console to figure out fc1 input size
ResWidth = 200
k_conv = 2
s_conv = 1
p_conv = 1
k_pool = 2
s_pool = 2
conv_out_channels = 16

conv1outWidth = (ResWidth - k_conv + 2*p_conv)/s_conv + 1
poolOutWidth = (conv1outWidth - k_pool)/s_pool+1
fc1Input = conv_out_channels*poolOutWidth**2
print(fc1Input)   #need to update this
########################################################################################
                #
              #
            #
            #
          #
        #
        #
      #
    #
    #
  #
#
# ### Model with multiple layers
# class SimpleCNN(nn.Module):
#     # make a note to understand what each of these inputs are.
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
#         #Channel = 3 is for RGB, Out channels
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(160000,60)
#         self.fc2 = nn.Linear(60, 60)
#         self.fc3 = nn.Linear(60,1)# Output a single scalar value

#     def forward(self, x):
#         x = self.conv1(x)
#       #  print("After Conv1:",x.size())
#         x = self.relu(x)
#         x = self.pool(x)
#       #  print("After pooling:",x.size())
#         x = x.reshape(x.size(0),-1) 
#        # print("After flattening:",x.size())
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x


### Model with multiple layers
class SimpleCNN(nn.Module):
    # make a note to understand what each of these inputs are.
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        #Channel = 3 is for RGB, Out channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(40000,300)
        self.fc2 = nn.Linear(300, 1)


    def forward(self, x):
        x = self.conv1(x)
      #  print("After Conv1:",x.size())
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
      #  print("After pooling:",x.size())
        x = x.reshape(x.size(0),-1) 
       # print("After flattening:",x.size())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the SimpleCNN model
model = SimpleCNN()
model = model.to(device)
# Define a loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)


############################# DATA #####################################################################
def create_loader(train_dataset, test_dataset, batch_size):
    torch.manual_seed(0) # For reproduciblity of random number in PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Creates a training DataLoader from this Dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size) # Creates a testing DataLoader from this Dataset
    return train_loader, test_loader


full_image_array_t = np.load('full_image_train.npy')
full_target_array_t = np.load('full_target_train.npy')

images_tensor_t = torch.tensor(full_image_array_t, dtype=torch.float).to(device)
images_tensor_t = images_tensor_t.permute(0, 3, 1, 2)
images_tensor_t = images_tensor_t.to(device)
targets_tensor_t = torch.tensor(full_target_array_t, dtype=torch.float).to(device)

full_image_array_v = np.load('full_image_val.npy')
full_target_array_v = np.load('full_target_val.npy')

images_tensor_v = torch.tensor(full_image_array_v, dtype=torch.float).to(device)
images_tensor_v = images_tensor_v.permute(0, 3, 1, 2)
images_tensor_v = images_tensor_v.to(device)
targets_tensor_v = torch.tensor(full_target_array_v, dtype=torch.float).to(device)

#Noramlization

images_tensor_t /= 255
images_tensor_v /= 255


dataset_t = TensorDataset(images_tensor_t, targets_tensor_t)
dataset_v = TensorDataset(images_tensor_v, targets_tensor_v)

train_loader, val_loader = create_loader(dataset_t, dataset_v, 5)


# Training loop
epochs = 10000  # Number of training epochs

training_losses = []
validation_losses = []
epochsarray = []
    
# for epoch in range(epochs):
#     # Zero the gradients
#     model.train()
#     optimizer.zero_grad()

#     # Forward pass
#     outputs = model(images_tensor_t)  # Pass the batch of images

#     # Compute the loss for the batch
#     train_loss = criterion(outputs, targets_tensor_t)  # Pass the batch of targets
    
#     # Backpropagation and optimization
#     train_loss.backward()
#     optimizer.step()
train_loss = np.zeros([epochs])
val_loss = np.zeros([epochs])

for epoch in range(epochs):
        model.train() # put model in training mode
        loss_tr_batch = [] # append this list to keep track of the loss
        # iterate over training set
        for _, data_tr in enumerate(train_loader): # go through mini-batches
            x_tr_batch,y_tr_batch = data_tr # read mini-batches
            y_tr_batch = y_tr_batch.type(torch.float)
            out_tr = model(x_tr_batch) # output of the model
            loss_tr = criterion(out_tr,y_tr_batch) # compute the cost
            loss_tr_batch.append(loss_tr.item()) # append the loss
            
            optimizer.zero_grad() # set gradients to zero
            loss_tr.backward() # backpropagation
            optimizer.step() # update the weights
           
            #_, yhat_tr_batch = torch.max(out_tr.data, 1) # find the predicted class
            #total_tr += y_tr_batch.size(0) # accumulate the number of samples
            #correct_tr += (yhat_tr_batch == y_tr_batch).sum().item() # accumulate the number of correct predictions
            
        train_loss[epoch] = np.mean(loss_tr_batch) # Compute average loss over epoch
        #loss_tr_accuracy[epoch] = 100*correct_tr/total_tr

        # Now let's keep track of the testing error without it influencing the training process
        model.eval() # put model in evaluation mode
        correct_te = 0 # initialize error counter
        total_te = 0 # initialize total counter
        loss_te_batch = [] # append this list to keep track of the loss
        with torch.no_grad(): # Make sure the gradients are 'off' during the testing process
            for _, data_te in enumerate(val_loader): # go through mini-batches
                x_te_batch, y_te_batch = data_te # read mini-batches
                y_te_batch = y_te_batch.type(torch.long)
                out_te = model(x_te_batch) # output of the model
                loss_te_batch.append(criterion(out_te,y_te_batch).item()) # append the loss
                #_, yhat_te_batch = torch.max(out_te.data, 1) # find the predicted class
                #total_te += y_te_batch.size(0) # accumulate the number of samples
                #correct_te += (yhat_te_batch == y_te_batch).sum().item() # accumulate the number of correct predictions
                
        val_loss[epoch] = np.mean(loss_te_batch) # Compute average loss over epoch
        #loss_te_accuracy[epoch] = 100*correct_te/total_te # Compute accuracy over epoch
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {train_loss[epoch]}")
        print(f"Validation Loss: {val_loss[epoch]}")
        training_losses.append(train_loss[epoch])
        validation_losses.append(val_loss[epoch])
        epochsarray.append(epoch)
    #
    #
      #
        #
        #
          #
            #
            #
              #
                #
    # model.eval()
    # val_outputs = model(images_tensor_v)
    # val_loss = criterion(val_outputs, targets_tensor_v)
                #
              #
            #
            #
          #
        #
        #
      #
    #
    #
  #
#        
        # print(f"Epoch [{epoch+1}/{epochs}]")
        # print(f"Training Loss: {train_loss.item()}")
        # print(f"Validation Loss: {val_loss.item()}")
        # training_losses.append(train_loss.item())
        # validation_losses.append(val_loss.item())
        # epochsarray.append(epoch)

# To make predictions on new data, use the model like this:
# model.eval() #setting model to eval mode so we can analyze without adjusting weights and biases
# new_image = torch.randn(1, 3, 200, 200)  # Replace with your new image
# new_image = new_image.to(device)
# prediction = model(new_image)
# print(f'Prediction: {prediction.item()}')






fig, ax = plt.subplots(figsize=(6,4.5),dpi=300)

ax.semilogy(epochsarray, training_losses,  color='#BB0000', label='Training loss',zorder=2)
ax.semilogy(epochsarray, validation_losses,  color='#347900', label='Validation loss',zorder=3)

plt.legend(loc='best')
plt.minorticks_on()
plt.grid(True, zorder=0)
plt.grid(which='major', color = 'black', alpha=0.5, linewidth='0.5')
plt.grid(which='minor', linestyle='--', linewidth='0.5', alpha=0.5)
plt.xlabel('Epochs')
plt.ylabel('Loss value')

plt.show()
