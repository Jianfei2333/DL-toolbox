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

def getdata(transform, kwargs={}):
  """
  norm1:
    mean = [0.6678, 0.5298, 0.5244]
    var = [0.2527, 0.1408, 0.1364]
  """
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
  train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind), **kwargs)
  val_dataloader = DataLoader(valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_ind), **kwargs)

  print ("Collect data complete!\n")

  return {
    'train': train_dataloader,
    'val': val_dataloader
  }