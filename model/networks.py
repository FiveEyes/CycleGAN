import torch
import torchvision
from torch import nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torchvision import transforms, utils 
import os


def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)

def conv_block(in_channels, out_channels, kernel_size, stride, padding, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )

def convt_block(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
    def forward(self, x):
        #return F.relu(self.shortcut(x) + self.model(x))
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, n_blocks=6, use_bias=False):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv_block(in_channels, dim, 7, stride=1, padding=0),
            conv_block(dim, dim*2, 3, stride=2, padding=1, bias=use_bias),
            conv_block(dim*2, dim*4, 3, stride=2, padding=1, bias=use_bias),

            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),

            convt_block(dim*4, dim*2, 3, stride=2,padding=1, output_padding=1, bias=use_bias),
            convt_block(dim*2, dim, 3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channels, 7, stride=1, padding=0),
            nn.Tanh()

        )

        init_weights(self.model)
        
    def forward(self, x):
        return self.model(x)



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

            nn.Conv2d(dim*2, dim*4, kw, stride=2, padding=(1,2), bias=False),
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*4, dim*8, kw, stride=1, padding=(2,1), bias=False),
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2, True),

            #nn.Conv2d(dim*8, dim*8, kw, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(dim*8),
            #nn.LeakyReLU(0.2, True),

            nn.Conv2d(dim*8, 1, kw, stride=1, padding=1),
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
        #if target_is_real:
        #    return -pred.mean()
        #else:
        #    return pred.mean()
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(pred)
        #print("pred, target:", pred.shape, target_tensor.shape)
        return self.loss(pred, target_tensor)

def calculate_gradient_penalty(netD, real_images, fake_images):
    batch_size = real_images.size(0)
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                            prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
    return grad_penalty
        