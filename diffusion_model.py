import torch
from torch import nn
import numpy as np
class Block(nn.module):
    def __init__(self,in_channels, out_channels, time_emb_dim):
        super(Block, self).__init__()
        self.time =  nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, timestep):
        shortcut = self.shortcut(x)
        t = self.relu(self.time(timestep))
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = x + t
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)

class downStep(nn.Module):
  def __init__(self, in_channels, out_channels, timestep):
    super(downStep, self).__init__()
    #todo
    self.maxp = nn.Sequential(nn.MaxPool2d(2),
        Block(in_channels, out_channels, timestep)
        )

  def forward(self, x, timestep):
    #todo
    down1 = self.maxp(x, timestep)
    # t = self.relu(self.time(timestep))
    # down1 = down1 + t
    return down1


class upStep(nn.Module):
  def __init__(self, in_channels, out_channels, timestep):
    super(upStep, self).__init__()
    #todo
    self.expanding = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,stride = 2)#Upsample(scale_factor = 2, mode='bilinear'
    self.c = Block(in_channels, out_channels, timestep)

  def forward(self, x, y, timestep):
    #todo
    x1 = self.expanding(x)
    crop_x = (y.size()[2] - x1.size()[2]) // 2
    crop_y = (y.size()[3] - x1.size()[3]) // 2

    y = y[:,:,crop_x:y.size()[2] - crop_x,crop_y:y.size()[3] - crop_y] # 12: 48, 12:48

    blk = torch.cat([y,x1], dim=1)
    output = self.c(blk, timestep)
    #t = self.relu(self.time(timestep))
    #output = output + t
    return output

class Diffusion_model(nn.module):
    def __init__(self):
        timestep = 32
        self.time = nn.Sequential(
            Sine_Cosine_Embedding(timestep),
            nn.Linear(timestep, timestep),
            nn.ReLU()
        )
        self.c1 = Block(1,64, 1)
        self.d1 = downStep(64, 128, timestep)
        self.d2 = downStep(128, 256, timestep)
        self.d3 = downStep(256, 512, timestep)
        self.d4 = downStep(512,1024, timestep)
        self.u1 = upStep(1024, 512, timestep)
        self.u2 = upStep(512, 256, timestep)
        self.u3 = upStep(256, 128, timestep)
        self.u4 = upStep(128, 64, timestep)
        self.c2 = nn.Conv2d(64, 1, kernel_size=1) 

    def forward(self, x, timestep):
        t = self.time(timestep)
        y = self.c1(x, t)
        
        l1 = self.d1(y, t)
        
        l2 = self.d2(l1, t)
        
        l3 = self.d3(l2, t)
        
        l4 = self.d4(l3, t)
        
        l6 = self.u1(l4, l3, t)
        
        l7 = self.u2(l6, l2, t)
        
        l8 = self.u3(l7, l1, t)
        
        l9 = self.u4(l8, y, t)
        
        out = self.c2(l9)

        return out



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