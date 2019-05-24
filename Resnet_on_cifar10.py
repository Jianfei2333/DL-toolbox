# Global environment setup.
import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

# Essential network building blocks.
from Networks import ResBlock as R
from Networks import Flatten

# Data loader.
from DataUtils import cifar10

# Useful tools.
from tools import train_and_check as mtool

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim

# Training setup.
os.environ['print_every'] = '10'
os.environ['save_every'] = '10'
TRAIN_EPOCHS=200
LEARNING_RATE=0.1

# GOT DATA
train_dataloader, val_dataloader, test_dataloader, sample = cifar10.getdata()

# DEFINE MODEL
model = nn.Sequential(
  nn.Conv2d(3, 64, (3,3), stride=1, padding=1),
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
  nn.AvgPool2d((4,4)),
  Flatten.Layer(),
  nn.Linear(512, 10)
)

# DEFINE OPTIMIZER
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# RUN TRAINING PROCEDURE
mtool.train(model, optimizer, train_dataloader, val_dataloader, TRAIN_EPOCHS)