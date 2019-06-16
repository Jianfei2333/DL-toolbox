from PIL import Image
import os
import sys
import torchvision.transforms as T
import numpy as np
sys.path.append('/home/huihui/Project/DL-toolbox/')
import glob

from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)
globalconfig.update_parser_params(args)

# os.environ['datapath'] = os.environ['datapath'].replace('_with_Color_Constancy', '')
os.environ['datapath'] = '/home/huihui/Data/ISIC2019_resize_crop/'

from DataUtils import isic2018 as data
imgsize = (1024, 1024)
# imgcount = 10015

dataloader = data.getdata({'train': T.ToTensor(), 'val': None})

dset = dataloader[0]['train'].dataset

imgcount = dset.__len__()
print ('Length: {}'.format(dset.__len__()))

"""
ISIC2018:
  mean = [0.76352127,0.54612797,0.57053038]
  var = [0.01994079,0.02337621,0.02901369]
  std = [0.14121186,0.15289281,0.17033405]
ISIC2018_with_Color_Constancy:
  mean = [0.62488488,0.62468347,0.62499634]
  var = [0.01315181,0.02681948,0.02968089]
  std = [0.11468134,0.16376653,0.17228143]
ISIC2019_resize
  mean = [0.5722533, 0.57208944, 0.57222467]
  var = [0.0363115, 0.04595451, 0.05087139]
  std = [0.19055578, 0.21437002, 0.22554687]
ISIC2019_resize_crop
  mean = [0.56935831, 0.56919221, 0.56929657]
  var = [0.03586197, 0.04661301, 0.05185368]
  std = [0.18937256, 0.21590046, 0.22771403]
"""

def computeMean():
  pixels = imgsize[0] * imgsize[1] * imgcount
  print (pixels)
  running_mean = np.array([0.,0.,0.])
  for k in range(dset.__len__()):
    img = dset.__getitem__(k)
    img = img[0].numpy()
    running_mean += np.sum(img, axis=(1,2)).astype(np.float)/pixels
    print ('After img', k, 'Mean:', running_mean)
  print ('Total mean:', running_mean)

# computeMean()

def computeVar():
  pixels = imgsize[0] * imgsize[1] * imgcount
  running_val = np.array([0.,0.,0.])
  mean = np.array([0.56935831, 0.56919221, 0.56929657])
  for k in range(dset.__len__()):
    img = dset.__getitem__(k)
    img = img[0].numpy()
    running_val += np.sum((img - mean[:, None, None]) ** 2, axis=(1,2)).astype(np.float)/pixels
    print ('After img', k, 'Mean:', running_val)
  print ('Total val:', running_val)
  print ('Total std:', np.sqrt(running_val))

computeVar()