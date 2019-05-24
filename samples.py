# Global environment setup.
import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

# Essential network building blocks.
from Networks import TwoLayerFC
from Networks import ThreeLayerConvNet
from Networks import LinearReLU

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
os.environ['save_every'] = '1'
TRAIN_EPOCHS=20
LEARNING_RATE=0.1

# GOT DATA
train_dataloader, val_dataloader, test_dataloader, sample = cifar10.getdata()

# DEFINE MODEL

# 通过nn.Sequential方式创建Module
# model = nn.Sequential(
#   LinearReLU.Model(3*32*32, 4000),
#   LinearReLU.Model(4000, 1000),
#   nn.Linear(1000, 10)
# )

# 通过直接创建方式创建Module
model = ThreeLayerConvNet.Model(3, 32, 16, 10)

# DEFINE OPTIMIZER
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# RUN TRAINING PROCEDURE
mtool.train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, 10)