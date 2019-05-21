import torch

import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from utils.utils import flatten

class Model(nn.Module):
  """
  The linear ReLU block.
  Structure:
    FC - ReLU
  
  Attributes:
    input_size: The dimension of input data, e.g. 32*32*3 for CIFAR-10.
    output_size: The dimension of output size, for the most time, the number of classes this classifier has.
  """
  def __init__(self, input_size, output_size):
    super().__init__()
    self.fc = nn.Linear(input_size, output_size)
    nn.init.kaiming_normal_(self.fc.weight)

  def forward(self, x):
    x = flatten(x)
    scores = F.relu(self.fc(x))
    return scores

def test_LinearReLU():
  input_size = 50
  x = torch.zeros((64, input_size), dtype=torch.float32)
  model = Model(input_size, 42)
  scores = model(x)
  print (scores.size())