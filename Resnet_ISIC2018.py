# Global environment setup.
import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

# Essential network building blocks.
from Networks import ResBlock as R
from Networks import LinearReLU
from Networks import Flatten

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
TRAIN_EPOCHS=50
LEARNING_RATE=0.1

# 设置从头训练/继续训练
continue_train=False
PRETRAIN_EPOCHS=0

pretrain_model_path = os.environ['savepath']+str(PRETRAIN_EPOCHS)+'epochs.pkl'
step=0

# GOT DATA
train_dataloader, val_dataloader, test_dataloader, sample, weights = isic2018.getdata()


# DEFINE MODEL
model = nn.Sequential(
  nn.Conv2d(3, 64, (7,7), stride=2, padding=3),
  nn.AvgPool2d((2,2)),
  R.Model(64, 64),
  R.Model(64, 64),
  R.Model(64, 64),
  R.Model(64, 128, downsample=True),
  R.Model(128, 128),
  R.Model(128, 128),
  R.Model(128, 128),
  R.Model(128, 256, downsample=True),
  R.Model(256, 256),
  R.Model(256, 256),
  R.Model(256, 256),
  R.Model(256, 256),
  R.Model(256, 256),
  R.Model(256, 512, downsample=True),
  R.Model(512, 512),
  R.Model(512, 512),
  nn.AvgPool2d((7,7)),
  Flatten.Layer(),
  LinearReLU.Model(512, 1000),
  nn.Linear(1000,7)
)

if continue_train:
  model_checkpoint = torch.load(pretrain_model_path)
  model.load_state_dict(model_checkpoint['state_dict'])
  print('Checkpoint restored!')
  step = model_checkpoint['episodes']
  os.environ['logdir'] = model_checkpoint['logdir']


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