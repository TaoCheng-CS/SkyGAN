import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class LSUN(Dataset):
    def __init__(self, path_root, imageSize):
        super(LSUN, self).__init__()
        self.path_root=path_root
        self.img_list= os.listdir(path_root)
        self.__len__= len(self.img_list)
        self.imageSize = imageSize
        self.transform=transforms.Compose([
                               transforms.Resize(self.imageSize),
                               transforms.CenterCrop(self.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ])
        self.item = []
        
    def get_data(self):
        print("Load data form disk to ram:")
        for i in tqdm(range(self.__len__)):
            x=Image.open(self.path_root+self.img_list[i])
            self.item.append(self.transform(x))
        
    def __len__(self):
        return self.__len__
        
    def __getitem__(self, index):
        return self.item[index]