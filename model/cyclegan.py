import torch
import torchvision
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils 
import os
import itertools
from model.networks import Generator, Discriminator, CycleGANLoss, calculate_gradient_penalty
from model.utils import ItemPool
class CycleGAN:
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.netG_A = Generator()
        self.netG_B = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()
        self.D_loss = CycleGANLoss()
        self.id_loss = nn.L1Loss()
        self.rec_loss = nn.L1Loss()
        self.pool_A = ItemPool()
        self.pool_B = ItemPool()

        self.nets = [self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.D_loss, self.id_loss, self.rec_loss]
        for net in self.nets:
            net.cuda()

        self.lambda_rec = 10.0
        self.lambda_id = 0.1
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = 0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr = 0.0002, betas=(0.5, 0.999))

    def forward(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
        self.fake_B = self.netG_A(self.real_A)
        self.fake_A = self.netG_B(self.real_B)

    
    def backward_D(self, retain_graph=False):
        real_A_D = self.netD_A(self.real_A)
        real_B_D = self.netD_B(self.real_B)
        fake_A = Variable(torch.Tensor(self.pool_A(self.fake_A.detach().cpu().numpy()))).cuda()
        fake_B = Variable(torch.Tensor(self.pool_B(self.fake_B.detach().cpu().numpy()))).cuda()
        fake_A_D = self.netD_A(fake_A)
        fake_B_D = self.netD_B(fake_B)
        
        self.loss_D_A = self.D_loss(real_A_D, True) + self.D_loss(fake_A_D, False) #+ calculate_gradient_penalty(self.netD_A, self.real_A, fake_A)
        #self.loss_D_A.backward(retain_graph=retain_graph)

        self.loss_D_B = self.D_loss(real_B_D, True) + self.D_loss(fake_B_D, False) #+ calculate_gradient_penalty(self.netD_B, self.real_B, fake_B)
        #self.loss_D_B.backward(retain_graph=retain_graph)

        self.loss_D = (self.loss_D_A + self.loss_D_B)

        self.loss_D.backward(retain_graph=retain_graph)

        return self.loss_D


    def backward_G(self, retain_graph=False):
        real_A = self.real_A
        real_B = self.real_B
        fake_A = self.fake_A
        fake_B = self.fake_B
        rec_A  = self.netG_B(fake_B)
        rec_B  = self.netG_A(fake_A)
        fake_A_D = self.netD_A(fake_A)
        fake_B_D = self.netD_B(fake_B)
        #id_A   = self.netG_B(real_A)
        #id_B   = self.netG_A(real_B)
        #self.loss_id = (self.id_loss(id_A, real_A) + self.id_loss(id_B, real_B))
        self.loss_rec = self.rec_loss(rec_A, real_A) + self.rec_loss(rec_B, real_B)
        self.loss_G_A = self.D_loss(fake_B_D, True)
        self.loss_G_B = self.D_loss(fake_A_D, True)
        
        self.loss_G = (self.loss_G_A + self.loss_G_B) * 10.0 + self.loss_rec #+ self.loss_id
        self.loss_G.backward(retain_graph=retain_graph)

        #self.loss_G_A = self.D_loss(fake_B_D, True) + self.id_loss(id_B, real_B) * 5.0 + self.rec_loss(rec_B, real_B) * 10.0
        #self.loss_G_A.backward(retain_graph=retain_graph)

        #self.loss_G_B = self.D_loss(fake_A_D, True) + self.id_loss(id_A, real_A) * 5.0 + self.rec_loss(rec_A, real_A) * 10.0
        #self.loss_G_B.backward(retain_graph=retain_graph)

        return self.loss_G

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, real_A, real_B):
        self.forward(real_A, real_B)

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


        return self.loss_G, self.loss_D

    def save_model(self, path):
        torch.save(self.netG_A.state_dict(), path + 'G_A.pt')
        torch.save(self.netG_B.state_dict(), path + 'G_B.pt')
        torch.save(self.netD_A.state_dict(), path + 'D_A.pt')
        torch.save(self.netD_B.state_dict(), path + 'D_B.pt')
    def load_model(self, path):
        if not os.path.exists(path + 'G_A.pt'):
            print('no model file exists')
            return
        print('loading model...', end='')
        self.netG_A.load_state_dict(torch.load(path + 'G_A.pt'))
        self.netG_B.load_state_dict(torch.load(path + 'G_B.pt'))
        self.netD_A.load_state_dict(torch.load(path + 'D_A.pt'))
        self.netD_B.load_state_dict(torch.load(path + 'D_B.pt'))
        print('done')
        for net in self.nets:
            net.cuda()