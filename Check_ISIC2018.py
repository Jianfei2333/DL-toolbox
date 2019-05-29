import os
import torch
import torch.nn as nn
import numpy as np
from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)

from Networks import ResBlock as R
from Networks import Flatten
from Networks import LinearReLU
from DataUtils import isic2018
from tools import metrics

# Model to check.
modelpath = '/data0/jianfei/models/Resnet34_ISIC2018/220epochs.pkl'

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

model.load_state_dict(torch.load(modelpath)['state_dict'])

_1, _2, test_dataloader, _3, _4 = isic2018.getdata()

def check(loader, model, step=0):
  """
  Check the accuracy of the model on validation set / test set.

  Args:
    loader: A Torchvision DataLoader with data to check.
    model: A PyTorch Module giving the model to check.

  Return:
    Nothing, but print the accuracy result to console.
  """
  print ("Checking on test dataset.")

  device='cuda:0'

  model = model.to(device=device)
  model.eval()
  scores = None
  y_truth = None
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      score = model(x)
      score = score.cpu()
      if scores is None:
        scores = score.numpy()
      else:
        scores = np.vstack((scores, score.numpy()))

      if y_truth is None:
        y_truth = y.numpy()
      else:
        y_truth = np.hstack((y_truth, y.numpy()))
  print(scores.shape)
  print(y_truth.shape)
  scoring = metrics.isic18(scores, y_truth, np.array(['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']))
  print (scoring)


check(test_dataloader, model)
      