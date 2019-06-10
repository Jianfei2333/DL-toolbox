# Global environment setup.
import os
# Arg parser
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)
globalconfig.update_parser_params(args)

# Essential network building blocks.
from Networks.Nets import Resnet
from torchvision import models

# Data loader.
import torchvision.transforms as T
from DataUtils import isic2018 as data

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 下面开始进行主干内容

transform = {
  'train': T.Compose([
    T.Resize((600,600)), # 放大
    T.RandomResizedCrop((224,224)), # 随机裁剪后resize
    T.RandomHorizontalFlip(0.5), # 随机水平翻转
    T.RandomVerticalFlip(0.5), # 随机垂直翻转
    T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
    T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
    T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
    T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
    T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
    T.ToTensor(),
    T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
  ]), 
  'val': T.Compose([
    T.Resize((224,224)), # 放大
    T.CenterCrop((224,224)),
    # T.RandomResizedCrop((224,224)), # 随机裁剪后resize
    # T.RandomHorizontalFlip(0.5), # 随机水平翻转
    # T.RandomVerticalFlip(0.5), # 随机垂直翻转
    # T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
    # T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
    # T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
    # T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
    # T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
    T.ToTensor(),
    T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
  ])
}

# GOT DATA
dataloader = data.getdata(transform)

# DEFINE MODEL
model = models.resnet152(pretrained=True)
# Modify.
num_fcin = model.fc.in_features
model.fc = nn.Linear(num_fcin, len(dataloader['train'].dataset.classes))

# print (model)

if args['continue']:
  model = globalconfig.loadmodel(model)

print ('Params to learn:')
params_to_update = []
for name,param in model.named_parameters():
  print('name:', name)
  if param.requires_grad == True:
    params_to_update.append(param)
    print ('\t', name)

# DEFINE OPTIMIZER
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(params_to_update, lr=args['learning_rate'])

criterion = nn.functional.cross_entropy

# Useful tools.
from tools import train_and_check as mtool

# RUN TRAINING PROCEDURE
mtool.train(
  model,
  dataloader,
  optimizer,
  criterion,
  args['epochs']
)