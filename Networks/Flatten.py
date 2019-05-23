import torch
import torch.nn as nn
from Networks.Utils.utils import flatten

class Layer(nn.Module):
  """
  Flatten as a layer.
  """
  def forward(self, x):
    return flatten(x)

def test_Flatten():
  x = torch.zeros((64, 3, 32, 32), dtype=torch.float32)
  model = Layer()
  scores = model(x)
  print (scores.size())