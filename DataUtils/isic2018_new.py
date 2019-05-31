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

  transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
  ])

  data = dset.ImageFolder(datapath+'Data', transform=transform)
  
  labels = np.array(data.imgs)[:, 1]
  C = len(data.classes)

  # weights: The number of images in different classes
  weights = np.zeros(C)
  for i in range(C):
    weights[i] = np.where(labels == str(i), 1, 0).sum()
  data.weights = weights

  batch = int(os.environ['batch-size'])
  train_arr = np.array(np.load(datapath+'train.npy'), dtype='int')
  val_arr = np.array(np.load(datapath+'validation.npy'), dtype='int')

  train_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_arr))
  val_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_arr))

  print ("Collect data complete!\n")

  return (train_dataloader, val_dataloader, weights)