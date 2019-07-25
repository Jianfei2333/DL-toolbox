import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torch

import torchvision.transforms as T
import torchvision.datasets as dset

datapath = os.environ['datapath'].replace('ISIC2018-openset', 'support')

def getdata(transform={'train':None, 'val':None}, kwargs={'num_workers': 20, 'pin_memory': True}):

  print ("Collecting data ...")

  traindata = dset.ImageFolder(datapath+'Data', transform=transform['train'])
  train4valdata = dset.ImageFolder(datapath+'Data', transform=transform['val'])
  
  size = traindata.__len__()
  
  labels = np.array(traindata.imgs)[:, 1]
  C = len(traindata.classes)

  # weights: The number of images in different classes
  weights = np.zeros(C)
  for i in range(C):
    weights[i] = np.where(labels == str(i), 1, 0).sum()
  traindata.weights = weights

  batch = int(os.environ['batch-size'])

  dataloader_5folds = [0,0,0,0,0]
  for k in range(5):
    train_ind = list(range(size))
    train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind), **kwargs)
    train4val_dataloader = DataLoader(train4valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind), **kwargs)
    val_dataloader = None
    dataloader_5folds[k] = {
      'train': train_dataloader,
      'train4val': train4val_dataloader,
      'val': val_dataloader
    }

  print ("Collect data complete!\n")

  return dataloader_5folds
