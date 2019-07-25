from PIL import Image
import os
import sys
import torchvision.transforms as T
import numpy as np
sys.path.append('/home/huihui/Project/DL-toolbox/')
import glob
import json

from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run(args, False)

os.environ['datapath'] = '/data0/Data/support/'

from DataUtils import ImgFolder as data
imgsize = (600, 450)
# imgcount = 10015

dataloader = data.getdata({'train': T.ToTensor(), 'val': None})

dset = dataloader[0]['train'].dataset

imgcount = dset.__len__()
print ('Length: {}'.format(dset.__len__()))

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
  return running_mean

# computeMean()

def computeVar(mean):
  pixels = imgsize[0] * imgsize[1] * imgcount
  running_var = np.array([0.,0.,0.])
  # mean = np.array([0.56935831, 0.56919221, 0.56929657])
  mean = mean
  for k in range(dset.__len__()):
    img = dset.__getitem__(k)
    img = img[0].numpy()
    running_var += np.sum((img - mean[:, None, None]) ** 2, axis=(1,2)).astype(np.float)/pixels
    print ('After img', k, 'Var:', running_var)
  print ('Total var:', running_var)
  print ('Total std:', np.sqrt(running_var))
  return (running_var, np.sqrt(running_var))

# computeVar()

def generateDatainfo():
  mean = computeMean()
  var, std = computeVar(mean)
  res = {
    'mean': mean.tolist(),
    'std': std.tolist(),
    'classes': dset.classes
  }
  with open(os.environ['datapath']+'info.json', 'w') as outfile:
    json.dump(res, outfile, indent=2)
  print (res)
  print ('Over!')

generateDatainfo()