import torch
from torch import nn
import numpy as np

class Block(nn.module):
    def __init__(self):
        pass

    def forward(self):
        pass

class Diffusion_model(nn.module):
    def __init__(self):
        pass

    def forward(self):
        pass

class Sine_Cosine_Embedding(nn.module):
    def __init__(self, dim):
        self.dim = dim
        self.n = 10000
    
    def forward(self, timesteps):
        # Expecting timesteps as [[0],[1],[2],[3],[4],....,[T]]
        i = len(timesteps)
        denominator = self.n**((2*i)//self.dim)
        timesteps = timesteps/denominator
        return torch.hstack(torch.sin(timesteps), torch.cos(timesteps))