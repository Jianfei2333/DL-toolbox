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

def getdata(type, transform={'train':None, 'val':None}, kwargs={'num_workers': 4, 'pin_memory': True}):

  print ("Collecting data ...")

  data = dset.ImageFolder(datapath+type, transform=transform['val'])

  batch = int(os.environ['batch-size'])

  loader = DataLoader(data, batch_size=batch, **kwargs)
  print ("Collect data complete!\n")

  return loader