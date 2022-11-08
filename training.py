import torch
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create loss function object
criterion = ...

# Optimizer function
optimizer  = Adam(...)

# define training loop parameters
num_epochs = ...

# training loop
for epoch in range(num_epochs):
    pass