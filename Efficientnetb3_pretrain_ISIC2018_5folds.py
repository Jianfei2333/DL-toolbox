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
from Networks.Losses import RecallWeightedCrossEntropy
from torchvision import models
from efficientnet_pytorch import EfficientNet

# Data loader.
import torchvision.transforms as T
from DataUtils import isic2018 as data

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 下面开始进行主干内容

from tools import datainfo
info = datainfo.getdatainfo()
mean = info['mean']
std = info['std']
normalize = T.Normalize(mean=mean, std=std)

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
    # T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
    normalize
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
    # T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
    normalize
  ])
}

# GOT DATA
dataloaders = data.getdata(transform)

# DEFINE MODEL
models = [None, None, None, None, None]
for i in range(5):
  models[i] = EfficientNet.from_pretrained('efficientnet-b3')
  # Modify.
  num_fcin = models[i]._fc.in_features
  models[i]._fc = nn.Linear(num_fcin, len(info.classes))

# print (model)

if args['continue']:
  models = globalconfig.loadmodels(models)
else:
  for i in range(5):
    models[i].step=0
    models[i].epochs=0

params = []
for i in range(5):
  models[i] = models[i].to(device=os.environ['device'])
  params_to_update = []
  for name,param in models[i].named_parameters():
    if param.requires_grad == True:
      params_to_update.append(param)
  params.append(params_to_update)

# DEFINE OPTIMIZER
optimizers = [None, None, None, None, None]
for i in range(5):
  optimizers[i] = optim.SGD(params[i], lr=args['learning_rate'], momentum=0.9)
  # optimizer = optim.Adam(params[i], lr=args['learning_rate'])

criterion = nn.functional.cross_entropy
# criterion = RecallWeightedCrossEntropy.recall_cross_entropy


# Useful tools.
from tools import train_and_check as mtool

# RUN TRAINING PROCEDURE
# mtool.train(
#   model,
#   dataloader[0],
#   optimizer,
#   criterion,
#   args['epochs']
# )

mtool.train5folds(
  models,
  dataloaders,
  optimizers,
  criterion,
  args['epochs']
)