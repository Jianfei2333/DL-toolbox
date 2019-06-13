from Networks.Blocks import Flatten
from Networks.Blocks import LinearReLU
from Networks.Blocks import ResBlock
from Networks.Nets import ThreeLayerConvNet
from Networks.Nets import TwoLayerFC

# Configuration
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)
globalconfig.update_parser_params(args)

# Flatten.test_Flatten()
# LinearReLU.test_LinearReLU()
# ResBlock.test_ResBlock()
# ThreeLayerConvNet.test_ThreeLayerConvNet()
# TwoLayerFC.test_TwoLayerFC()

###
# Color constancy test samples.
###

def check_color_constancy():
  import numpy as np
  import os
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  from PIL import Image

  from tools import colorConstancy
  import torchvision.transforms as T
  from DataUtils import isic2018 as data
  # transform = {
  #   'train': T.Compose([
  #     T.ToTensor()
  #   ]),
  #   'val': T.Compose([
  #     T.ToTensor()
  #   ])
  # }
  dataloader = data.getdata({'train':None, 'val':None}, kwargs={'num_workers':0, 'pin_memory':False})
  dset = dataloader['train'].dataset
  
  for i in np.arange(0, dset.__len__(), 5):
    print (dset.__getitem__(i))
    methods = 3
    img = [0,0,0,0,0]
    img1 = [0,0,0,0,0]
    img2 = [0,0,0,0,0]
    img3 = [0,0,0,0,0]
    img4 = [0,0,0,0,0]
    for j in range(5):
      img1[j] = np.asarray(dset.__getitem__(i+j)[0])
      # img1[j] = np.transpose(img[j], (1,2,0))
      img2[j] = colorConstancy.Grey_world(img1[j])
      img3[j] = colorConstancy.His_equ(img1[j])
      # img4[j] = colorConstancy.White_balance(img1[j])
    for k in range(5):
      plt.subplot(methods,5,k+1)
      plt.title('Original')
      plt.imshow(img1[k])
      plt.axis('off')

      plt.subplot(methods,5,k+6)
      plt.title('Grey World')
      plt.imshow(img2[k])
      plt.axis('off')

      plt.subplot(methods,5,k+11)
      plt.title('Hist Equlize')
      plt.imshow(img3[k])
      plt.axis('off')

      # plt.subplot(methods,5,k+16)
      # plt.title('White Balance')
      # plt.imshow(img4[k] - img1[k])
      # plt.axis('off')

    plt.show()


check_color_constancy()