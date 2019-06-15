import torch
import torchvision
from torch import nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils 
import os

def same_padding(kernel_size):
    if type(kernel_size) is not tuple:
        return (kernel_size - 1) // 2
    else:
        return tuple([ (ks - 1) // 2 for ks in kernel_size ])


def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.model(x)

class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False):
        super(ConvTBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = same_padding(kernel_size)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0, bias=False)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            #nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return F.relu(self.shortcut(x) + self.model(x))


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, n_blocks=6, use_bias=False):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            ConvBlock(in_channels, dim, 7, stride=1, padding=0),
            ConvBlock(dim, dim*2, 3, stride=2, padding=1, bias=use_bias),
            ConvBlock(dim*2, dim*4, 3, stride=2, padding=1, bias=use_bias)
        )

        transformer = []
        for i in range(n_blocks):
            transformer.append(ResidualBlock(dim * 4, dim * 4, 3))
        self.transformer = nn.Sequential(*transformer)

        self.decoder = nn.Sequential(
            ConvTBlock(dim*4, dim*2, 3, stride=2,padding=1, output_padding=1, bias=use_bias),
            ConvTBlock(dim*2, dim, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channels, 7, stride=1, padding=0),
            nn.Tanh()
        )
        init_weights(self.encoder)
        init_weights(self.transformer)
        init_weights(self.decoder)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.transformer(out)
        #print("out.shape", out.shape)
        out = self.decoder(out)
        return out



class Discriminator(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super(Discriminator, self).__init__()
        kw = 4
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, dim, kw, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim, dim*2, kw, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*2, dim*4, kw, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*4, dim*8, kw, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*8, dim*8, kw, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*8, 1, kw, stride=1, padding=1, bias=True),
        )
        init_weights(self.model)
    def forward(self, x):
        return self.model(x)
    

class CycleGANLoss(nn.Module):
    def __init__(self,):
        super(CycleGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0).cuda())
        self.register_buffer('fake_label', torch.tensor(0.0).cuda())
        self.loss = nn.MSELoss()
    def __call__(self, pred, target_is_real):
        if target_is_real:
            return -pred.mean()
        else:
            return pred.mean()
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(pred)
        #print("pred, target:", pred.shape, target_tensor.shape)
        return self.loss(pred, target_tensor)