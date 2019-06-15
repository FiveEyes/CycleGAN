from model import cyclegan
from data import image_folder
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

batch_sz = 4
test_sz = 10
tfs = transforms.Compose([
    #transforms.Resize((128,128)), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def save_tensor_img(x, path):
    image = x.cpu().detach().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1) / 2.0 * 255.0
    image = image.astype(np.uint8)
    #print(image)
    image = Image.fromarray(image)
    image.save(path)

test_A = DataLoader(
    image_folder.ImageFolder('datasets/horse2zebra/testA', transform = transforms.ToTensor()),
    batch_size = 1, shuffle=True)
test_B = DataLoader(
    image_folder.ImageFolder('datasets/horse2zebra/testB', transform = transforms.ToTensor()),
    batch_size = 1, shuffle=True)

def gen_test_fake(model):
    print('generating...', end='')

    for i, (real_A, real_B) in enumerate(zip(test_A, test_B)):
        if i >= test_sz:
            break
        fake_B = model.netG_A(real_A.cuda())
        fake_A = model.netG_B(real_B.cuda())
        #print(fake_B[0].shape, fake_A[0].shape)
        save_tensor_img(fake_B[0], './fakeB/' + str(i) + '.png')
        save_tensor_img(fake_A[0], './fakeA/' + str(i) + '.png')
    print('done.')

if __name__ == '__main__':
    dataset_A = image_folder.ImageFolder('datasets/horse2zebra/trainA', transform = tfs)
    dataloader_A = DataLoader(dataset_A, batch_size = batch_sz, shuffle=True)
    dataset_B = image_folder.ImageFolder('datasets/horse2zebra/trainB', transform = tfs)
    dataloader_B = DataLoader(dataset_A, batch_size = batch_sz, shuffle=True)


    print(len(dataloader_A), len(dataloader_B))
    n = len(dataloader_A)
    model = cyclegan.CycleGAN()
    model.load_model('')

    
    for epoch in range(100):
        print("epoch:", epoch)
        i = 0
        total_loss_G = 0.0
        total_loss_D = 0.0
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            #print(real_A[0])
            loss_G, loss_D = model.train_step(real_A.cuda(),real_B.cuda())
            loss_G = loss_G.cpu().detach().numpy()
            loss_D = loss_D.cpu().detach().numpy()
            total_loss_G += loss_G
            total_loss_D += loss_D
            s = str(i) + '/' +  str(n) +  ' ' + str(loss_G) + ' ' + str(loss_D)
            sys.stdout.write('%s\r' % s)
            i += 1
        print("loss:", total_loss_G / n, total_loss_D / n)
        model.save_model('')
        
        gen_test_fake(model)