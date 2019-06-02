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

  transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.6678, 0.5298, 0.5244), std=(0.2527, 0.1408, 0.1364))
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
  train_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind))
  val_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_ind))

  print ("Collect data complete!\n")

  return (train_dataloader, val_dataloader, weights)