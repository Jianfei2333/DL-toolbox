from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import sys
sys.path.append('../')

DATAPATH = './data'
NUM_TRAIN = 49000

def getdata():
  
  print ("Collecting data...")

  transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

  cifar10_train = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  sample = cifar10_train.__getitem__(0)[0][None, :, :, :]
  train_dataloader = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

  cifar10_val = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  val_dataloader = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

  cifar10_test = dset.CIFAR10(DATAPATH, train=True, download=True, transform=transform)
  test_dataloader = DataLoader(cifar10_test, batch_size=64)

  print ("Collect data complete!\n")

  return (train_dataloader, val_dataloader, test_dataloader, sample)
