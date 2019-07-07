import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torch
import copy

import torchvision.transforms as T
import torchvision.datasets as dset

datapath = os.environ['datapath']

def getdata(transform={'train':None, 'val':None}, unknown=[0,3], kwargs={'num_workers': 20, 'pin_memory': True}):

  print ("Collecting data ...")

  traindata = dset.ImageFolder(datapath+'Data', transform=transform['train'])
  train4valdata = dset.ImageFolder(datapath+'Data', transform=transform['val'])
  valdata = dset.ImageFolder(datapath+'Data', transform=transform['val'])
  
  ind_mapper = {}
  i = 1
  classes = len(traindata.classes)
  for j in range(classes):
    if j in unknown:
      ind_mapper[j] = 0
    else:
      ind_mapper[j] = i
      i += 1

  t = 0
  unknown_idxs = np.array([])
  for img, label in traindata:
    traindata.targets[t] = ind_mapper[label]
    train4valdata.targets[t] = ind_mapper[label]
    valdata.targets[t] = ind_mapper[label]
    if label in unknown:
      unknown_idxs = np.hstack((unknown_idxs, t))
    t += 1

  c2i = copy.deepcopy(traindata.class_to_idx)
  for k in traindata.class_to_idx:
    c2i[k] = ind_mapper[c2i[k]]
    if traindata.class_to_idx[k] in unknown:
      del(c2i[k])
  c2i['unknown'] = 0
  traindata.class_to_idx = c2i
  traindata.classes = list(c2i.keys())
  train4valdata.class_to_idx = c2i
  train4valdata.classes = list(c2i.keys())
  valdata.class_to_idx = c2i
  valdata.classes = list(c2i.keys())

  labels = np.array(traindata.imgs)[:, 1]
  C = len(traindata.classes)

  # weights: The number of images in different classes
  weights = np.zeros(C)
  for i in range(C):
    weights[i] = np.where(labels == str(i), 1, 0).sum()
  traindata.weights = weights
  valdata.weights = weights

  batch = int(os.environ['batch-size'])

  fold_ind = [np.array([]) for i in range(5)]
  for _ in range(5):
    fold_ind[_] = np.array(np.load(datapath+str(_)+'fold.npy'), dtype='int')

  dataloader_5folds = [0,0,0,0,0]
  for k in range(5):
    VAL_IND = 4-k
    train_ind = np.array([])
    val_ind = np.array([])
    for i in range(5):
      if i == VAL_IND:
        val_ind = fold_ind[i]
      else:
        train_ind = np.hstack((train_ind, fold_ind[i])).astype('int')

    to_move_ind = np.intersect1d(train_ind, unknown_idxs)
    train_ind = np.setdiff1d(train_ind, to_move_ind)
    val_ind = np.union1d(val_ind, to_move_ind)

    train_dataloader = DataLoader(traindata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind), **kwargs)
    train4val_dataloader = DataLoader(train4valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind), **kwargs)
    val_dataloader = DataLoader(valdata, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_ind), **kwargs)
    dataloader_5folds[k] = {
      'train': train_dataloader,
      'train4val': train4val_dataloader,
      'val': val_dataloader
    }

  print ("Collect data complete!\n")

  return dataloader_5folds