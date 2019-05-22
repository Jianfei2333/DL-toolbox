import torch.nn as nn
from utils.utils import flatten

class Layer(nn.Module):
  """
  Flatten as a layer.
  """
  def forward(self, x):
    return flatten(x)