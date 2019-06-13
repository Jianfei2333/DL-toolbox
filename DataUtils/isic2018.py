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

def getdata(transform={'train':None, 'val':None}, kwargs={'num_workers': 4, 'pin_memory': True}):

  print ("Collecting data ...")

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

  train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_arr), **kwargs)
  val_dataloader = DataLoader(valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_arr), **kwargs)

  print ("Collect data complete!\n")

  return {
    'train': train_dataloader,
    'val': val_dataloader
  }