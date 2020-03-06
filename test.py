from glob import glob
import os
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

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
output_num = 5
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




model = Net()
model.load_state_dict(torch.load('weight.pkl'))
images = cv2.imread('train/0/0_454.png')
true_num=0
false_num=0
y_test =np.array([],dtype=int)
y_pred =np.array([],dtype=int)
for i, data in enumerate(test_loader):
    #print(data[0].size())  # input image
    #print(data[1]) 
    print(data[0].size())
    predict = model(data[0]).argmax()
    real = data[1][0]
    if(predict == real):
       true_num = true_num+1
    else:
       false_num = false_num+1
    y_test = np.append(y_test,np.array(real))
    y_pred = np.append(y_pred,np.array(predict))

    print("predict : "+str(predict)+"\treal :"+str(real)+"\t true : "+str(true_num)+"\t false : "+str(false_num)+"\t percent : "+str(true_num/(true_num+false_num)))



def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):
   if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
   else:
      print('Confusion matrix, without normalization')
   print(cm)
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout()

cnf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['1', '2', '3', '4', '5']

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
title='Normalized confusion matrix')

plt.show()
