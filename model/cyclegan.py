import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils 
import os
import itertools
from model.networks import Generator, Discriminator, CycleGANLoss

class CycleGAN:
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.netG_A = Generator()
        self.netG_B = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()
        self.nets = [self.netG_A, self.netG_B, self.netD_A, self.netD_B]
        for net in self.nets:
            net.cuda()
        self.D_loss = CycleGANLoss()
        self.id_loss = nn.L1Loss()
        self.rec_loss = nn.L1Loss()
        self.lambda_rec = 10.0
        self.lambda_id = 0.5
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = 0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = 0.0002, betas=(0.5, 0.999))

    def forward(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B
        self.fake_B = self.netG_A(self.real_A)
        #print(self.real_A.shape, self.fake_B.shape)
        self.id_B   = self.netG_A(self.real_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_A  = self.netG_B(self.fake_B)
        self.id_A   = self.netG_B(self.real_A)
        self.rec_B  = self.netG_A(self.fake_A)

        self.real_A_D = self.netD_A(self.real_A)
        self.fake_A_D = self.netD_A(self.fake_A)

        self.real_B_D = self.netD_B(self.real_B)
        self.fake_B_D = self.netD_B(self.fake_B)

    def backward_D_helper(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.D_loss(pred_real, True)

        pred_fake = netD(fake)
        loss_D_fake = self.D_loss(pred_fake, True)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    def backward_D(self, retain_graph=False):
        #self.loss_D_A = self.backward_D_helper(self.D_A, self.real_A, self.fake_A)
        #self.loss_D_B = self.backward_D_helper(self.D_B, self.real_B, self.fake_B)
        #self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D_A = self.D_loss(self.real_A_D, True) + self.D_loss(self.fake_A_D, False)
        self.loss_D_B = self.D_loss(self.real_B_D, True) + self.D_loss(self.fake_B_D, False)
        self.loss_D = (self.loss_D_A + self.loss_D_B) * 0.5
        self.loss_D.backward(retain_graph=retain_graph)
        return self.loss_D
        
    def backward_G(self, retain_graph=False):
        self.loss_id = (self.id_loss(self.id_A, self.real_A) + self.id_loss(self.id_B, self.real_B))

        self.loss_rec =  (self.rec_loss(self.rec_A, self.real_A) + self.rec_loss(self.rec_B, self.real_B)) * 10.0
        
        self.loss_G_A = self.D_loss(self.fake_A_D, True)
        self.loss_G_B = self.D_loss(self.fake_B_D, True)
        
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_id + self.loss_rec
        
        self.loss_G.backward(retain_graph=retain_graph)
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

        self.set_requires_grad([self.netG_A, self.netG_B], False)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D(retain_graph=True)
        self.optimizer_D.step()

        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G(retain_graph=False)
        self.optimizer_G.step()

        
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
        self.nets = [self.netG_A, self.netG_B, self.netD_A, self.netD_B]
        for net in self.nets:
            net.cuda()