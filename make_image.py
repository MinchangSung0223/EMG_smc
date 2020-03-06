import csv
import numpy as np
import pandas
import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

filename = 'TotalRing.csv'
ch_len = 8
n_classes = 5


dataset = pandas.read_csv(filename, header=None)
dataset = dataset.values
print(dataset.shape)
X = np.array(dataset[:,0:ch_len]).astype(float)
Y = np.array(dataset[:,ch_len]).astype(int)
dir_path="/home/sung/EMG_test/Ring"
os.mkdir("."+"/train/")
os.mkdir("."+"/test/")
for n in range(0,n_classes):
   os.mkdir("."+"/train/"+str(n)+"/")
   os.mkdir("."+"/test/"+str(n)+"/")
X_len = len(X)

Train_len = int(X_len*0.75)
Test_len = X_len-Train_len

temp1 = np.ones(X_len)
temp1[0:Test_len] = 0
np.random.shuffle(temp1)
permute_num =temp1.copy()
print(permute_num)


img = np.zeros((ch_len,ch_len),dtype="float")
img_label = np.zeros((ch_len,1),dtype="int")
for i in range(0,X_len-1):
   temp=img[0:ch_len-1,:].copy()
   img[0,:] = (X[i,:]-X[i,:].min())/(X[i,:].max()-X[i,:].min())
   img[1:ch_len,:] = temp;

   temp_label = img_label[0:ch_len-1].copy()
   img_label[0] = Y[i]
   img_label[1:ch_len] = temp_label;
   save_label =int(np.round(np.mean(img_label)))


   #normalize
   img_temp = img*255
   img_ = img_temp.astype("uint8")
   #print(img_)
   #save png
   foldername="train"
   if(permute_num[i]==0):
      foldername="test"
   img_name = str(save_label)+"_"+str(i)+".png"
   os.chdir(dir_path+"/"+foldername+"/"+str(save_label))
   cv2.imwrite(img_name, img_)



