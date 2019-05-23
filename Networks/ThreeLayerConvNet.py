import torch

import torch.nn as nn
import torch.nn.functional as F

import random

import sys
sys.path.append("..")
from Networks.Utils.utils import flatten

class Model(nn.Module):
  """
  Three layers plain convolutional neural network.
  Structure:
    Conv - ReLU - Conv - ReLU - FC

  Attributes:
    in_channel: The number of channels in the input image
    channel_1: The number of channels of the hidden layer #1
    channel_2: The number of channels of the hidden layer #2
    out_dim: The output size of fully connected layer #3
  """
  def __init__(self, in_channel, channel_1, channel_2, out_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channel, channel_1, (5,5), stride=1, padding=2)
    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.constant_(self.conv1.bias, 0)

    self.conv2 = nn.Conv2d(channel_1, channel_2, (3,3), stride=1, padding=1)
    nn.init.kaiming_normal_(self.conv2.weight)
    nn.init.constant_(self.conv2.bias, 0)

    self.fc = nn.Linear(channel_2 * 32*32, out_dim)
    nn.init.kaiming_normal_(self.fc.weight)
    nn.init.constant_(self.fc.bias, 0)

  def forward(self, x):
    a = F.relu(self.conv1(x))
    b = F.relu(self.conv2(a))
    b_flat = flatten(b)
    scores = self.fc(b_flat)

    return scores

def test_ThreeLayerConvNet():
  x = torch.zeros((64, 3, 32, 32), dtype=torch.float32)
  channel1 = random.randint(0, 100)
  channel2 = random.randint(0, 100)
  model = Model(in_channel=3, channel_1=channel1, channel_2=channel2, out_dim=10)
  scores = model(x)
  print(scores.size())