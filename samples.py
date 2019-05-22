from networks import TwoLayerFC
from networks import ThreeLayerConvNet
from networks import LinearReLU
from DataUtils import cifar10
from tools import train_and_check as mtool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter(log_dir='/data0/jianfei/tensorboard-log')

train_dataloader, val_dataloader, test_dataloader = cifar10.getdata()

# 通过nn.Sequential方式创建Module
model = nn.Sequential(
  LinearReLU.Model(3*32*32, 4000),
  LinearReLU.Model(4000, 1000),
  nn.Linear(1000, 10)
)

# 通过直接创建方式创建Module
#model = ThreeLayerConvNet.Model(3, 32, 16, 10)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=1e-2)

images, labels = next(iter(train_dataloader))

grid = torchvision.utils.make_grid(images)
writer.add_text('seq', 'This is a sequential network.', 0)
writer.close()

# 调用训练函数
mtool.train(model, optimizer, train_dataloader, val_dataloader, 10)
