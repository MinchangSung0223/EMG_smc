from glob import glob
import os
import numpy as np
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



import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
ch_len = 8
output_num = 5
if torch.cuda.is_available():
    is_cuda = True

class EMGDataset(Dataset):
   def __init__(self,):
      self.files = glob(".")
   def __len__(self):
      pass
   def __getitem__(self,idx):
      pass
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
test_loader = DataLoader(valid ,batch_size=10,shuffle=True,num_workers=16)
train_loader= DataLoader(train ,batch_size=10,shuffle=True,num_workers=16)

#valid_data_gen = torch.utils.data.DataLoader(valid,batch_size=64,num_workers=3)
#dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
#dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
#dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
def plot_img(image):
     image = image.numpy()[0]
     mean = 0.1307
     std = 0.3081
     image = ((mean * image)+std)
     plt.imshow(image,cmap='gray')
sample_data = next(iter(train_loader))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(64, 16) # cv
        self.fc2 = nn.Linear(16, output_num)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)


def fit(epoch,model,data_loader,phase='training',volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx , (data,target) in enumerate(data_loader):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        data , target = Variable(data,volatile),Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        
        running_loss += F.nll_loss(output,target,size_average=False).data
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss.item()/len(data_loader.dataset)
    accuracy = 100.0 * running_correct.item()/len(data_loader.dataset)
    
    return loss,accuracy

model = Net()
if is_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(),lr=0.01)
data , target = next(iter(train_loader))
if is_cuda:
    data=data.cuda()
output = model(Variable(data))
output.size()
target.size()

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,100):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    print("EPOCH : "+str(epoch)+"\t"+"LOSS : "+str(epoch_loss)+"\t"+"ACCURACY : "+str(epoch_accuracy))
    torch.save(model.state_dict(), 'weight.pkl')


plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
