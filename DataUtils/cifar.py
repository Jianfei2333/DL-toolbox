import os
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

DATAPATH='/home/huihui/Data/cifar10/'

def getdata(unkown=[]):
  
  print ("Collecting data...")

  transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

  cifar10_train = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  
  train_dataloader = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

  cifar10_val = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  val_dataloader = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

  print ("Collect data complete!\n")

  return ()

getdata()
