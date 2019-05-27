import torch.nn as nn
from Networks import ResBlock as R
from Networks import Flatten
from Networks import LinearReLU

def Resnet34():
  return nn.Sequential(
    nn.Conv2d(3, 64, (7,7), stride=2, padding=3),
    nn.MaxPool2d((3,3), stride=2),
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
    # nn.Linear(512, 1000) # OVER
    LinearReLU.Model(512, 1000),
    nn.Linear(1000,7)
  )

def Resnet50():
  return nn.Sequential(
    nn.Conv2d(3, 64, (7,7), stride=2, padding=3),
    nn.MaxPool2d((3,3), stride=2),
    R.BottleNeck(64, 64, 256),
    R.BottleNeck(256, 64, 256),
    R.BottleNeck(256, 64, 256),
    R.BottleNeck(256, 128, 512, downsample=True),
    R.BottleNeck(512, 128, 512),
    R.BottleNeck(512, 128, 512),
    R.BottleNeck(512, 128, 512),
    R.BottleNeck(512, 256, 1024, downsample=True),
    R.BottleNeck(1024, 256, 1024),
    R.BottleNeck(1024, 256, 1024),
    R.BottleNeck(1024, 256, 1024),
    R.BottleNeck(1024, 256, 1024),
    R.BottleNeck(1024, 256, 1024),
    R.BottleNeck(1024, 512, 2048, downsample=True),
    R.BottleNeck(2048, 512, 2048),
    R.BottleNeck(2048, 512, 2048),
    nn.AvgPool2d((7,7)),
    Flatten.Layer(),
    # nn.Linear(2048, 1000) # OVER
    LinearReLU.Model(2048, 1000), 
    nn.Linear(1000,7)
  )