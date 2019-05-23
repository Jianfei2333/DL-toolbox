import os
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)


from Networks import ResBlock as R
from Networks import Flatten

from DataUtils import cifar10

from tools import train_and_check as mtool

import torch
import torch.nn as nn
import torch.optim as optim

os.environ['print_every'] = '10'
TRAIN_EPOCHS=500

train_dataloader, val_dataloader, test_dataloader = cifar10.getdata()


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

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

mtool.train(model, optimizer, train_dataloader, val_dataloader, TRAIN_EPOCHS)

mtool.checkAcc(test_dataloader, model, TRAIN_EPOCHS)

torch.save(model.state_dict(), os.environ['filename'] + '-' + str(TRAIN_EPOCHS) + 'epochs.pkl')