import torch

import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from Networks.Utils.utils import flatten

class Model(nn.Module):
  """
  Two Layer Fully Connected Network.
  Structure:
    FC - ReLU - FC
  
  Attributes:
    input_size: The dimension of input data, e.g. 32*32*3 for CIFAR-10.
    hidden_size: The dimension of hidden layer.
    output_size: The dimension of output size, for the most time, the number of classes this classifier has.
  """
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    nn.init.kaiming_normal_(self.fc1.weight)
    self.fc2 = nn.Linear(hidden_size, output_size)
    nn.init.kaiming_normal_(self.fc2.weight)

  def forward(self, x):
    x = flatten(x)
    scores = self.fc2(F.relu(self.fc1(x)))
    return scores

def test_TwoLayerFC():
  input_size = 50
  x = torch.zeros((64, input_size), dtype=torch.float32)
  model = Model(input_size, 42, 10)
  scores = model(x)
  print (scores.size())