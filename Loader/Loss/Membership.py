import torch
import torch.nn as nn
import os

def loss(input, target, unknown_ind, weight=None):
  l = 5
  c = input.shape[1] - 1
  sigmoid = 1 / (1 + torch.exp(-input))
  sigmoid[range(0, sigmoid.shape[0]), target] = 1 - sigmoid[range(0, sigmoid.shape[0]), target]
  sigmoid = (l / (c-1)) * sigmoid * sigmoid
  sigmoid[range(0, sigmoid.shape[0]), target] = ((c-1)/l) * sigmoid[range(0, sigmoid.shape[0]), target]
  sigmoid = torch.cat([sigmoid[:, :unknown_ind], sigmoid[:, unknown_ind+1:]], dim=1)
  return torch.sum(sigmoid)

def prediction(input, unknown_ind):
  threshold = 0.5
  sigmoid = 1 / (1 + torch.exp(-input))
  val, ind = sigmoid.max(1)
  if int(os.environ['gpus']) == 0:
    device = 'cpu'
  elif int(os.environ['gpus']) == 1:
    device = os.environ['device']
  else:
    device = os.environ['device'][:6]
  pred = torch.where(val > threshold, ind, torch.tensor([unknown_ind]).to(device=device))
  return pred
