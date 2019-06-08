import torch
import torchvision.transforms as T
import time
import numpy as np
import os

from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run()
globalconfig.update_parser_params(args)

from DataUtils import isic2019 as data

print('Testing data load...')
prompt = ''
for num_workers in range(0, 50, 5):
  kwargs = {'num_workers': num_workers, 'pin_memory': False} if os.environ['device'] != 'cpu' else {}
  
  transform = {
    'train': T.Compose([
      T.Resize((500,500)), # 放大
      T.RandomResizedCrop((300,300)), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      T.Normalize(mean=(0.6678, 0.5298, 0.5244), std=(0.2527, 0.1408, 0.1364))
    ]), 
    'val': T.Compose([
      T.Resize((300,300)), # 放大
      T.CenterCrop((300,300)),
      # T.RandomResizedCrop((224,224)), # 随机裁剪后resize
      # T.RandomHorizontalFlip(0.5), # 随机水平翻转
      # T.RandomVerticalFlip(0.5), # 随机垂直翻转
      # T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      # T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      # T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      # T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      # T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      T.Normalize(mean=(0.6678, 0.5298, 0.5244), std=(0.2527, 0.1408, 0.1364))
    ])
  }

  start = time.time()
  for epoch in range(1):
    d = data.getdata(transform, kwargs)
    for batch_idx, (dat, target) in enumerate(d['train']):
      print("Epoch:{}, step:{}".format(epoch, batch_idx))
      pass
  end = time.time()
  p = "\nFinish with:{} second, num_workers={}".format(end-start, num_workers)
  prompt += p
  print(p)
  print('Finish one test.')
print (prompt)