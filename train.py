from model import cyclegan
from data import image_folder
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

batch_sz = 3

tfs = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

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