from glob import glob
import os
import pandas
import numpy as np
import itertools
import matplotlib.pyplot as plt
import shutil
from torchvision import datasets,transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
import sys
from sklearn.metrics import confusion_matrix

import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
output_num = 5
ch_len = 8
valid = dset.ImageFolder(root="test/",transform=transforms.Compose([
                               transforms.Scale(16),       # 한 축을 128로 조절하고
                               transforms.CenterCrop(16),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))

train = dset.ImageFolder(root="train/",transform=transforms.Compose([
                               transforms.Scale(16),       # 한 축을 128로 조절하고
                               transforms.CenterCrop(16),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))
test_loader = DataLoader(valid ,batch_size=1,shuffle=True,num_workers=8)
train_loader= DataLoader(train ,batch_size=1,shuffle=True,num_workers=8)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64, 16) # cv
        self.fc2 = nn.Linear(16, output_num )

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)



filename = 'TotalRing.csv'


dataset = pandas.read_csv(filename, header=None)
dataset = dataset.values
print(dataset.shape)
X = np.array(dataset[:,0:ch_len]).astype(float)
Y = np.array(dataset[:,ch_len]).astype(int)

print(X.shape)
X_len = len(X)
img = np.zeros((ch_len,ch_len),dtype="float")
img_label = np.zeros((ch_len,1),dtype="int")
for i  in range(1,X_len-ch_len):
   temp=img[0:ch_len-1,:].copy()
   img[0,:] = (X[i,:]-X[i,:].min())/(X[i,:].max()-X[i,:].min())
   img[1:ch_len,:] = temp;

   temp_label = img_label[0:ch_len-1].copy()
   img_label[0] = Y[i]
   img_label[1:ch_len] = temp_label;
   save_label =int(np.round(np.mean(img_label)))

   img_temp = img*255
   img_ = img_temp.astype("uint8")

   model = Net()
   model.load_state_dict(torch.load('weight.pkl'))
   #images = Image.open('test/0/0_3182.png')
   images = Image.fromarray(img_)
   images = images.convert('RGB')
   transform=transforms.Compose([
                               transforms.Scale(16),       # 한 축을 128로 조절하고
                               transforms.CenterCrop(16),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ])
   images = transform(images)
   images = images.view(1,3,16,16)
   print("\n\n")
   print(model(images).argmax())
   print(save_label)
   print("\n\n")

