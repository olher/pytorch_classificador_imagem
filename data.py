#%%
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
from PIL import Image

class data():

    def __init__(self):


        # LOAD DF
        self.cat = ['avião', 'automovel', 'passaro', 'gato', 'cervo'
                   ,'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

        to_tensor = T.Compose([T.Resize((32,32))
                              ,T.ToTensor()]) 

        self.cifar10_train = CIFAR10('train', train=True, download=True, transform=to_tensor)
        self.cifar10_test = CIFAR10('test', train=False, download=True, transform=to_tensor)

        imgs_train = torch.stack([tensor for tensor, _ in self.cifar10_train], dim=3)

        self.mean = tuple(imgs_train.view(3,-1).mean(dim=1).tolist())
        self.std = tuple(imgs_train.view(3,-1).std(dim=1).tolist())

        self.to_tensor = T.Compose([T.Resize((32,32))
                              ,T.ToTensor()
                              ,T.Normalize(mean=self.mean, std=self.std)])
        
        self.train = CIFAR10('train', train=True, download=False, transform=self.to_tensor)
        self.test = CIFAR10('test', train=False, download=False, transform=self.to_tensor)

        # DATALOADER
        self.train_dataloader = DataLoader(self.train, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(self.test, batch_size=64)


    def predict_data(self, path):
        
        self.img = Image.open(fp=(path)).convert('RGB')
        self.tensor = self.to_tensor(self.img)

        return self