import torch

import torch.nn as nn
import torch.nn.functional as F

import random

import sys
sys.path.append("..")
from utils.utils import flatten

class Model(nn.Module):
  """
  Deep residual network building block.
  Structure:
    x - Conv - ReLU - Conv - Add - ReLU
    |-------------------------------|

  Attributes:
    in_channel: The number of channels in input.
    out_channel: The number of channels in output.
    downsample: Boolean, use downsample or not.
  """
  def __init__(self, in_channel, out_channel, downsample=False):
    super().__init__()
    self.downsample = downsample
    if downsample:
      self.conv1 = nn.Conv2d(in_channel, out_channel, (3,3), stride=2, padding=1)
      self.conv_shortcut = nn.Conv2d(in_channel, out_channel, (1,1), stride=2, padding=0)
    else:
      self.conv1 = nn.Conv2d(in_channel, out_channel, (3,3), stride=1, padding=1)
    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.constant_(self.conv1.bias, 0)

    self.bn1 = nn.BatchNorm2d(out_channel)
    self.bn2 = nn.BatchNorm2d(out_channel)

    self.conv2 = nn.Conv2d(out_channel, out_channel, (3,3), stride=1, padding=1)
    nn.init.kaiming_normal_(self.conv2.weight)
    nn.init.constant_(self.conv2.bias, 0)

  def forward(self, x):

    a = F.relu(self.bn1(self.conv1(x)))
    b = self.bn2(self.conv2(a))
    if self.downsample:
      shortcut = self.conv_shortcut(x)
    else:
      shortcut = x
    scores = F.relu(b + shortcut)

    return scores

def test_ResBlock():
  x = torch.zeros((64, 128, 32, 32), dtype=torch.float32)
  outchannel = 128

  model = Model(in_channel=128,out_channel=outchannel, downsample=False)
  scores = model(x)
  print(scores.size())