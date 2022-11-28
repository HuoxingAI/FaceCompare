import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset

from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
 
# 添加椒盐噪声
class AddSaltPepperNoise(object):
 
    def __init__(self, density=0):
        self.density = density
 
    def __call__(self, img):
 
        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img

class my_dataset(Dataset):
    def __init__(self, root_dir,
                 data_list,
                 img_size=112,train=True):

        self.root_dir = root_dir
        random.shuffle(data_list)
        # self.sub_dirs = os.listdir(root_dir)
        self.data_list = data_list
        # self.label_dict = {}
        # for i in range(len(data_list)):
        #     # folder_name,label = df.iloc[i,0],df.iloc[i,1]
        #     self.label_dict[str(data_list[i][0])] = np.float(data_list[i][1])
        self.len = len(data_list)
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 缩放
            transforms.RandomHorizontalFlip(0.5),  # 左右镜像
            # transforms.RandomAffine(degrees=(-5, 5), translate=(5, 5),scale=(0.9, 1.1)),
            transforms.RandomPerspective(p=0.001),
            
            # transforms.GaussianBlur(kernel_size=5),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.1),saturation=(0.9, 1.1), hue=0.1),
            transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 缩放
            transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
        ])
        self.train = train


    def __len__(self):
        return self.len

    def __getitem__(self, index):

        # folder_name = self.sub_dirs[index]
        folder = os.path.join(self.root_dir,str(self.data_list[index][0]))
        files = os.listdir(folder)
        images = []
        np.random.shuffle(files)

        image_transforms = transforms.Compose([
            AddSaltPepperNoise(random.random()),
            # transforms.Grayscale(1),
            
        ])

        file_path0 = os.path.join(folder,files[0])
        img0 = Image.open(file_path0).convert("RGB")
        
        img0 = image_transforms(img0)
        if self.train:
            img0 = self.train_transform(img0)
        else:
            img0 = self.test_transform(img0)

        file_path1 = os.path.join(folder,files[1])
        img1 = Image.open(file_path1).convert("RGB")
        img1 = image_transforms(img1)
        if self.train:
            img1 = self.train_transform(img1)
        else:
            img1 = self.test_transform(img1)
        img0.sub_(0.5).div_(0.5) 
        img1.sub_(0.5).div_(0.5) 
        label = self.data_list[index][1]
        # print(img0)
        # print(img1)
        # if label == 0:
        #     label = -1
        return img0, img1,label


def my_data_loader(batch_size,root_dir, data_list,img_size=128,train=True):
    data_set = my_dataset(root_dir=root_dir,data_list=data_list,img_size=img_size,train=train)
    data_loader = DataLoader(data_set,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=1)
    return data_loader
