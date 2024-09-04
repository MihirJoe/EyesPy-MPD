# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:10:17 2023

@author: ian
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:08:43 2023

@author: campbell.1369
"""

""" trying timoshenko using the linear elastic example from 
Forward Problem for Plane Stress Linear Elasticity Boundary Value Problem
Weighted-Physics-Informed Neural Networks (W-PINNs)
Author: Alexandros D.L Papados
"""

import torch
import torch.nn as nn
import numpy as np
import time
from itertools import product
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123456)
np.random.seed(123456)

# E = 1                                       # Young's Modulus
# nu = 0.3                                    # Poisson Ratio
# G = ((E/(2*(1+nu))))                        # LEBVP coefficient

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()                                                  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(2, 20))                     # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              # First activation Layer
        for num in range(2, 4):                                                     # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(20, 20))       # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(20, 1))                 # Output Layer: two ourputs, u and v

    # Forward Feed
    def forward(self, x):
        return self.net(x)

        

    def loss(self, xy, sigx_sol):
        outs = self.net(xy)                             # Interior Solution
        sigx = outs                        # u and v interior
        
        
        loss_1 = (sigx - sigx_sol)
        
        loss = ((loss_1) ** 2).mean()
        
        return loss
    
""" This part will look different from example"""

#%%

# Problem setup:
    
L = 100
c = 5
P = 10000
P_tensor = P

nu = 0.3
E = 200*10**9 #steel
b = 1
G = 80*10**9 #steel
I = 2/3*b*c**3

    
x_values = np.linspace(0,L,num=100)
y_values = np.linspace(-c,c,num=int((len(x_values)*2*c/(L))))

coordinates = list(product(x_values, y_values))
coordinates_array = np.array(coordinates)

xy = coordinates_array
# x = coordinates_array[:,0]
# y = coordinates_array[:,1]

# xy = np.concatenate((x, y), axis=1)

xy_tensor = torch.tensor(xy,dtype=torch.float32)
xy_tensor = xy_tensor.to(device)

# x_tensor = torch.tensor(x,dtype=torch.float32)
# y_tensor = torch.tensor(y,dtype=torch.float32)

# x_tensor = x_tensor.view(-1, 1)
# y_tensor = y_tensor.view(-1, 1)


min_values, _ = torch.min(xy_tensor, dim=0)
max_values, _ = torch.max(xy_tensor, dim=0)

# Scale each column to the [-1, 1] range using min-max scaling
scaled_xy_tensor = 2 * (xy_tensor - min_values) / (max_values - min_values) - 1

scaled_xy_np = scaled_xy_tensor.detach().cpu().numpy()


#%%
""" Synthetic data"""

sigx_sol = -3/2*P/c**3*xy_tensor[:,0]*xy_tensor[:,1] #these might change if b wasnt 1?

# Calculate the mean and standard deviation of your target values
# mean_sigx_sol = torch.mean(sigx_sol)
# std_sigx_sol = torch.std(sigx_sol)

min_sigx_sol = torch.min(sigx_sol)
max_sigx_sol = torch.max(sigx_sol)


#map between -1 and 1
scaled_sigx_sol = 2*(sigx_sol - min_sigx_sol) / (max_sigx_sol - min_sigx_sol)-1

scaled_sigx_sol_np = scaled_sigx_sol.detach().cpu().numpy()
sigx_sol_np = sigx_sol.detach().cpu().numpy()

#%%
""" Back to example"""

model = Model().to(device)

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters())
epochs = 5000

# Training
def train(epoch):
    model.train()

    def closure():
        optimizer.zero_grad()
        loss = model.loss(scaled_xy_tensor, scaled_sigx_sol)

        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f"Epoch [{epoch}]")
    print(f"Loss: {loss.item()}")
    
#%%
""" training """
print('start training...')
tic = time.time()
for epoch in range(1, epochs + 1):
    train(epoch)
toc = time.time()
print(f'total training time: {toc - tic}')

#%%
""" The training data may need to be flattened, but then the test data 
which would be used here would be thye normal domain"""
scaled_stress = model(xy_tensor)
stress = (scaled_stress*(max_sigx_sol-min_sigx_sol)+min_sigx_sol)
stress_np = stress.detach().cpu().numpy()


plt.scatter(xy[:,0], xy[:,1], c=stress_np, cmap='viridis',label='Original Points', s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stress in x Visualization')
plt.legend()
plt.show()