import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

from Networks import TwoLayerFC
from Networks import ThreeLayerConvNet
from Networks import LinearReLU

from DataUtils import cifar10

from tools import train_and_check as mtool

import torch
import torch.nn as nn
import torch.optim as optim

# from Utils.globaltb import writer
# from Utils import visual as vis
# writer = writer()

os.environ['print_every'] = '10'

train_dataloader, val_dataloader, test_dataloader, sample = cifar10.getdata()

# 通过nn.Sequential方式创建Module
# model = nn.Sequential(
#   LinearReLU.Model(3*32*32, 4000),
#   LinearReLU.Model(4000, 1000),
#   nn.Linear(1000, 10)
# )

# 通过直接创建方式创建Module
model = ThreeLayerConvNet.Model(3, 32, 16, 10)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# 调用训练函数
mtool.train(model, optimizer, train_dataloader, val_dataloader, 10)
