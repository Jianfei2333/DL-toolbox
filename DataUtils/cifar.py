import os
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import copy

DATAPATH='/home/huihui/Data/cifar10/'

def getdata(unknown=[6,7,8,9], batch=64, transform=None):
  """
  getdata: get cifar-10 data with customed unknown classes.
  
    Args:
      - unknown: list, index of classes that is unknown.
      - batch: batch size.
      - transform: torchvision.transfrom.
    
    Returns:
      An object:
        - 'train': cifar-10 train dataloader.
        - 'val': cifar-10 validation dataloader.
        - 'train_sample': a sample dataloader for test running. A sample is conducted with 100 images each class (train & val total).
        - 'val_sample': a sample dataloader for test running.
  """
  print ("Collecting data...")

  sample_train = np.load(DATAPATH+'sample_train.npy')
  sample_val = np.load(DATAPATH+'sample_val.npy')
  train_ind = np.load(DATAPATH+'train.npy')
  val_ind = np.load(DATAPATH+'val.npy')
  
  if transform is None:
    transform = T.Compose([
      T.ToTensor(),
      T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

  data = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  t = 0
  unknown_idxs = np.array([])
  for img, label in data:
    if label in unknown:
      data.targets[t] = -1
      unknown_idxs = np.hstack((unknown_idxs, t))
    t += 1

  to_move_ind = np.intersect1d(train_ind, unknown_idxs)
  train_ind = np.setdiff1d(train_ind, to_move_ind)
  val_ind = np.union1d(val_ind, to_move_ind).astype('int')

  to_move_sample = np.intersect1d(sample_train, unknown_idxs)
  sample_train = np.setdiff1d(sample_train, to_move_sample)
  sample_val = np.union1d(sample_val, to_move_sample).astype('int')
  
  c2i = copy.deepcopy(data.class_to_idx)
  for k in data.class_to_idx:
    if data.class_to_idx[k] in unknown:
      del(c2i[k])
  c2i['unknown'] = -1
  data.class_to_idx = c2i
  data.classes = list(c2i.keys())

  train_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(train_ind))
  val_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(val_ind))
  train_sample_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(sample_train))
  val_sample_dataloader = DataLoader(data, batch_size=batch, sampler=sampler.SubsetRandomSampler(sample_val))

  print ("Collect data complete!\n")

  return {
    'train': train_dataloader,
    'val': val_dataloader,
    'train_sample': train_sample_dataloader,
    'val_sample': val_sample_dataloader
  }

getdata()