import torch
from torch import nn
import numpy as np
class ForwardDiffusion():

    def __init__(self, beta_start, beta_end, timesteps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.beta_schedule = self.linearBetaSchedule()
        self.alpha = self.alphaGeneration()
        self.alpha_bar = self.alphaBar()    
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0-self.alpha_bar)
    
    def alphaBar(self):
        return torch.cumprod(self.alpha, dim=0)

    def alphaGeneration(self):
        return 1.0-self.beta_schedule
    
    def linearBetaSchedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
    
    def forwardNoise(self, x, t):
        noise = torch.randn_like(x.float())
        alpha_root_bar =  torch.reshape(self.sqrt_alpha_bar[t], (-1, 1, 1, 1))
        one_min_alpha_root_bar = torch.reshape(self.sqrt_one_minus_alpha_bar[t], (-1, 1, 1, 1))
        noisy_image = ( alpha_root_bar * x ) + one_min_alpha_root_bar * noise
        return noisy_image, noise
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