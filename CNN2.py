
"""
Created on Mon Apr  8 17:06:45 2024

@author: campb
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

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(3)
        
        self.fc1 = nn.Linear(30000,1)
        

    def forward(self, x):
        
        x = self.conv1(x)       
        x = self.bn1(x)        
        x = self.relu(x)
        
        x = self.conv2(x)       
        x = self.bn2(x)        
        x = self.relu(x)       
        
        x = self.conv3(x)       
        x = self.bn3(x)        
        x = self.relu(x) 
        
        x = self.conv4(x) 
        x = self.bn4(x)        
        x = self.relu(x)

        x = self.pool(x)
        x = x.reshape(x.size(0),-1) 

        x = self.fc1(x)
        return x

# Create an instance of the SimpleCNN model
model = SimpleCNN()
model = model.to(device)
# Define a loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)


############################# DATA #####################################################################
def create_loader(train_dataset, test_dataset, batch_size):
    torch.manual_seed(0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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


batch_size = 12

dataset_t = TensorDataset(images_tensor_t, targets_tensor_t)
dataset_v = TensorDataset(images_tensor_v, targets_tensor_v)

train_loader, val_loader = create_loader(dataset_t, dataset_v, batch_size)


# Training loop
epochs = 3  # Number of training epochs

training_losses = []
validation_losses = []
epochsarray = []
    

train_loss = np.zeros([epochs])
val_loss = np.zeros([epochs])

for epoch in range(epochs):
        model.train() 
        loss_tr_batch = []

        for _, data_tr in enumerate(train_loader):
            x_tr_batch,y_tr_batch = data_tr
            y_tr_batch = y_tr_batch.type(torch.float)
            out_tr = model(x_tr_batch)
            loss_tr = criterion(out_tr,y_tr_batch)
            loss_tr_batch.append(loss_tr.item())
            
            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()
            
        train_loss[epoch] = np.mean(loss_tr_batch)


        model.eval() 
        loss_v_batch = [] 
        with torch.no_grad():
            for _, data_v in enumerate(val_loader):
                x_v_batch, y_v_batch = data_v 
                y_v_batch = y_v_batch.type(torch.long)
                out_v = model(x_v_batch) 
                loss_v_batch.append(criterion(out_v,y_v_batch).item())    
        val_loss[epoch] = np.mean(loss_v_batch) 

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {train_loss[epoch]}")
        print(f"Validation Loss: {val_loss[epoch]}")
        training_losses.append(train_loss[epoch])
        validation_losses.append(val_loss[epoch])
        epochsarray.append(epoch)


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







