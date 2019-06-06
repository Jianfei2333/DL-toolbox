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
from DataUtils import isic2019 as data

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim

# 下面开始进行主干内容

# GOT DATA
dataloader = data.getdata()

# DEFINE MODEL
model = models.resnet152(pretrained=True)
for param in model.layer3.parameters():
  # print (param)
  param.requires_grad = False
for param in model.layer4.parameters():
  # print (param)
  param.requires_grad = False 
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