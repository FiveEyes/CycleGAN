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
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

to_img_tf = transforms.ToPILImage('RGB')

def save_tensor_img(x, path):
    image = x.cpu().detach()
    image = image * 0.5 + 0.5
    image = to_img_tf(image)
    image.save(path)
    return
    image = x.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0)) * 127.5 + 127.5
    image = image.astype(np.uint8)
    #print(image)
    image = Image.fromarray(image)
    image.save(path)

test_A = DataLoader(
    image_folder.ImageFolder('datasets/horse2zebra/testA', transform = tfs),
    batch_size = 1, shuffle=True, num_workers=4)
test_B = DataLoader(
    image_folder.ImageFolder('datasets/horse2zebra/testB', transform = tfs),
    batch_size = 1, shuffle=True, num_workers=4)

def gen_test_fake(model):
    print('generating...', end='')

    for i, (real_A, real_B) in enumerate(zip(test_A, test_B)):
        if i >= test_sz:
            break
        fake_B = model.netG_A(real_A.cuda())
        rec_A = model.netG_B(fake_B)
        fake_A = model.netG_B(real_B.cuda())
        rec_B = model.netG_A(fake_A)

        #print(fake_B[0].shape, fake_A[0].shape)
        save_tensor_img(real_A[0], './fakeB/' + str(i) + '0.jpg')
        save_tensor_img(fake_B[0], './fakeB/' + str(i) + '1.jpg')
        save_tensor_img(rec_A[0], './fakeB/' + str(i) + '2.jpg')
        save_tensor_img(real_B[0], './fakeA/' + str(i) + '0.jpg')
        save_tensor_img(fake_A[0], './fakeA/' + str(i) + '1.jpg')
        save_tensor_img(rec_B[0], './fakeA/' + str(i) + '2.jpg')
    print('done.')

if __name__ == '__main__':
    dataset_A = image_folder.ImageFolder('datasets/horse2zebra/trainA', transform = tfs)
    dataloader_A = DataLoader(dataset_A, batch_size = batch_sz, shuffle=True, num_workers=4)
    dataset_B = image_folder.ImageFolder('datasets/horse2zebra/trainB', transform = tfs)
    dataloader_B = DataLoader(dataset_A, batch_size = batch_sz, shuffle=True, num_workers=4)


    print(len(dataloader_A), len(dataloader_B))
    n = len(dataloader_A)
    model = cyclegan.CycleGAN()
    model.load_model('')
    gen_test_fake(model)
    epoch = 0
    while True:
    #for epoch in range(100):
        
        print("epoch:", epoch)
        
        i = 0
        total_loss_G = 0.0
        total_loss_D = 0.0
        
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            #print(real_A[0].min(), real_A[0].max())
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
        
        epoch += 1