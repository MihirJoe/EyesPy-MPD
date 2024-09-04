# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:27:10 2023

@author: allen
"""
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a simple CNN architecture


class SimpleCNN(nn.Module):
    # make a note to understand what each of these inputs are.
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        """Channel = 3 is for RGB, Out channels - """
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(262144, 64)
        self.fc2 = nn.Linear(64, 1)  # Output a single scalar value

    def forward(self, x):
        x = self.conv1(x)
      #  print("After Conv1:",x.size())
        x = self.relu(x)
        x = self.pool(x)
      #  print("After pooling:",x.size())
        x = x.view(-1)
       # print("After flattening:",x.size())
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create an instance of the SimpleCNN model
model = SimpleCNN()

# Define a loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)
"make sure we are researching the Adam Optimizer b/c we should write about it"

# Assuming you have three images and their corresponding target values as tensors
# For example, images and targets are placeholders here, replace with your actual data
image_list = torch.load('C:/Users/allen/.spyder-py3/Images/DataLoad/images_tensor.pt')
"THIS WILL HAVE TO CHANGE FROM AN ABSOLUTE PATH TO A RELATIVE PATH"
#Relative Path: DataLoad/images_tensor.pt
#Absolute Path: C:/Users/allen/.spyder-py3/Images/DataLoad/images_tensor.pt
image_list = image_list.to(device)
image1 = image_list[0]
image1 = image1.to(torch.float32)
image2 = image_list[1]
image2 = image2.to(torch.float32)

target1 = torch.tensor(5.0)  # Replace with your actual target values
target2 = torch.tensor(5.0)
# making a loop that will be able to take a batch of images within a file and target values associated with each and create more of these tensors

# Training loop
epochs = 100  # Number of training epochs
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output1 = model(image1)  # Add a batch dimension
    output2 = model(image2)
    # research what unsqueeze means/batch learning
    # Compute the loss
    loss = criterion(output1, target1) + criterion(output2,target2)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# To make predictions on new data, use the model like this:
model.eval() #setting model to eval mode so we can analyze without adjusting weights and biases
new_image = torch.randn(3, 256, 256)  # Replace with your new image
prediction = model(new_image)
print(f'Prediction: {prediction.item()}')

