# Global environment setup.
import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

# Essential network building blocks.
from Networks import ResBlock as R
from Networks import LinearReLU
from Networks import Flatten
from Networks import Resnet

# Data loader.
# from DataUtils import cifar10
from DataUtils import isic2018

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim

# Training setup.
os.environ['print_every'] = '10'
os.environ['save_every'] = '10'
os.environ['batch-size'] = '64'
TRAIN_EPOCHS=50
LEARNING_RATE=1e-6

# 设置从头训练/继续训练
continue_train=False
PRETRAIN_EPOCHS=0

import sys
args = sys.argv[1:]
for arg in args:
  if arg.find('--learning-rate=') != -1:
    LEARNING_RATE=float(arg[16:])
    continue
  if arg.find('--batch-size=') != -1:
    os.environ['batch-size'] = arg[13:]
    continue
  if arg.find('--print-every=') != -1:
    os.environ['print_every='] = arg[14:]
    continue
  if arg.find('--save-every=') != -1:
    os.environ['save_every'] = arg[13:]
    continue
  if arg.find('--epochs=') != -1:
    TRAIN_EPOCHS = int(arg[9:])
    continue
  if arg.find('--continue') != -1:
    continue_train = True
    continue
  if arg.find('--pretrain=') != -1:
    PRETRAIN_EPOCHS = int(arg[11:])
    continue
  else:
    print('Args error!', arg, 'not found!')
    sys.exit()

pretrain_model_path = os.environ['savepath']+str(PRETRAIN_EPOCHS)+'epochs.pkl'
step=0

# 下面开始进行主干内容

# GOT DATA
train_dataloader, val_dataloader, test_dataloader, sample, weights = isic2018.getdata()

# DEFINE MODEL
model = Resnet.Resnet50()

if continue_train:
  model_checkpoint = torch.load(pretrain_model_path)
  model.load_state_dict(model_checkpoint['state_dict'])
  print('Checkpoint restored!')
  step = model_checkpoint['episodes']
  os.environ['tb-logdir'] = model_checkpoint['tb-logdir']


# DEFINE OPTIMIZER
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Useful tools.
from tools import train_and_check as mtool

# RUN TRAINING PROCEDURE
mtool.train(
  model,
  optimizer,
  train_dataloader,
  val_dataloader,
  test_dataloader,
  weights,
  PRETRAIN_EPOCHS,
  TRAIN_EPOCHS,
  step
)
