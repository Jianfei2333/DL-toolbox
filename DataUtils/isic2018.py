import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torch

import torchvision.transforms as T
import torchvision.datasets as dset

datapath = os.environ['datapath']

def getdata():

  print ("Collecting data ...")

  transform = {
    'train': T.Compose([
      T.Resize((600,600)), # 放大
      T.RandomResizedCrop((224,224)), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
    ]), 
    'val': T.Compose([
      T.Resize((224,224)), # 放大
      T.RandomResizedCrop((224,224)), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
    ])
  }

  traindata = dset.ImageFolder(datapath+'Data', transform=transform['train'])
  valdata = dset.ImageFolder(datapath+'Data', transform=transform['val'])
  
  labels = np.array(traindata.imgs)[:, 1]
  C = len(traindata.classes)

  # weights: The number of images in different classes
  weights = np.zeros(C)
  for i in range(C):
    weights[i] = np.where(labels == str(i), 1, 0).sum()
  traindata.weights = weights
  valdata.weights = weights

  batch = int(os.environ['batch-size'])
  train_arr = np.array(np.load(datapath+'train.npy'), dtype='int')
  val_arr = np.array(np.load(datapath+'validation.npy'), dtype='int')

  train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_arr))
  val_dataloader = DataLoader(valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_arr))

  print ("Collect data complete!\n")

  return {
    'train': train_dataloader,
    'val': val_dataloader
  }