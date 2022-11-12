import torch
from torch.optim import Adam
from torchvision import transforms

from diffusion_model import Diffusion_model
from dataloader import train_dataloader, test_dataloader

from matplotlib import pyplot as plt

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') # Modified to CPU 
print(device)

# Model Hyper parameters
beta_start = 0.0001
beta_end = 0.02
timesteps = 500

# Model
model = Diffusion_model(beta_start, beta_end, timesteps)
model = model.to(device)

# Create loss function object
criterion = ... # TODO: Define Loss function.

# Learning Rate
learning_rate = 1e-3

# Optimizer function
optimizer  = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

# define training loop parameters
num_epochs = 1

# Gray scale Tranform
transform = transforms.Compose([transforms.Grayscale()])

# training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    for _, data in enumerate(train_dataloader):
        color_images, _ = data
        gray_scale_images = transform(color_images)
        gray_scale_images = gray_scale_images.to(device)

        optimizer.zero_grad()
        output = model(gray_scale_images) # TODO: Pass required input to the model and get output.

        loss = criterion(...) # TODO: Add the required loss function with appropriate parameters.

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print("Epoch %d / %d and Loss: %d" % (epoch, num_epochs, epoch_loss))

# Testing
with torch.no_grad():
    for _, data in enumerate(test_dataloader):
        color_image, _ = data
        gray_scale_image = transform(color_image)

        output = model(gray_scale_images) # TODO: Provide required input to the model.

        # TODO: We can calculate the FID and IS scores here.

        # TODO: Visualize the outputs and inputs.