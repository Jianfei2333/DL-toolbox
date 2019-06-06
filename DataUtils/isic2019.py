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
  """
  norm1:
    mean = [0.6678, 0.5298, 0.5244]
    var = [0.2527, 0.1408, 0.1364]
  """
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
      T.Normalize(mean=(0.6678, 0.5298, 0.5244), std=(0.2527, 0.1408, 0.1364))
    ]), 
    'val': T.Compose([
      T.Resize((224,224)), # 放大
      T.CenterCrop((224,224)),
      # T.RandomResizedCrop((224,224)), # 随机裁剪后resize
      # T.RandomHorizontalFlip(0.5), # 随机水平翻转
      # T.RandomVerticalFlip(0.5), # 随机垂直翻转
      # T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      # T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      # T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      # T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      # T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      T.Normalize(mean=(0.6678, 0.5298, 0.5244), std=(0.2527, 0.1408, 0.1364))
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
  # Read fold infomation.
  fold_ind = [np.array([]) for i in range(5)]
  for _ in range(5):
    fold_ind[_] = np.array(np.load(datapath+str(_)+'fold.npy'), dtype='int')

  # f0-f3: train, f4: validation
  VAL_IND = 4
  train_ind = np.array([])
  val_ind = np.array([])
  for i in range(5):
    if i == VAL_IND:
      val_ind = fold_ind[i]
    else:
      train_ind = np.hstack((train_ind, fold_ind[i])).astype('int')
  train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind))
  val_dataloader = DataLoader(valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_ind))

  print ("Collect data complete!\n")

  return {
    'train': train_dataloader,
    'val': val_dataloader
  }