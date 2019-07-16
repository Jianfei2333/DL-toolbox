import torch
import torch.nn as nn
import os

# class DOCLoss(nn.Module):
#   def __init__(self):
#     super(DOCLoss, self).__init__()
#     # self.weight = weight

#   def forward(self, input, target, weight=None):
#     sigmoid = 1 / (1 + torch.exp(-input))
#     bias = torch.ones_like(input)
#     bias[range(0, bias.shape[0]), target] = 0
#     log_sigmoid = torch.log(bias - sigmoid)
#     loss = -torch.sum(log_sigmoid)
#     return loss

#   def prediction(input, t=0.5, weight=None):
#     sigmoid = 1 / (1 + torch.exp(-input))
#     values, indices = sigmoid.max(1)
#     predict = torch.where(values > t, indices, torch.tensor([-1]))
#     return predict

def loss(input, target, unknown_ind, weight=None):
  sigmoid = 1 - 1 / (1 + torch.exp(-input))
  sigmoid[range(0, sigmoid.shape[0]), target] = 1 - sigmoid[range(0, sigmoid.shape[0]), target]
  sigmoid = torch.log(sigmoid)
  sigmoid = torch.cat([sigmoid[:, :unknown_ind], sigmoid[:, unknown_ind+1:]], dim=1)
  weight = torch.cat([weight[:unknown_ind], weight[unknown_ind+1:]])/(1-weight[unknown_ind])
  if weight is not None:
    loss = -torch.sum(sigmoid * weight)
  else:
    loss = -torch.sum(sigmoid)
  return loss

def prediction(input, unknown_ind, t, weight=None):
  sigmoid = 1 / (1 + torch.exp(-input))
  # sigmoid = torch.cat([sigmoid[:, :unknown_ind], sigmoid[:, unknown_ind+1:]], dim=1)
  values, indices = sigmoid.max(1)
  if int(os.environ['gpus']) == 0:
    device = 'cpu'
  elif int(os.environ['gpus']) == 1:
    device = os.environ['device']
  else:
    device = os.environ['device'][:6]
  if t is None:
    import numpy as np
    t = torch.from_numpy(np.repeat(0.5, input.shape[1])).to(device=device)
  predict = torch.where(values > t[indices], indices, torch.tensor([unknown_ind]).to(device=device))
  return predict

def auto_threshold(y_true, scores, unknown_ind):
  sigmoid = 1 / (1 + torch.exp(-scores))
  prob = sigmoid[y_true] - 1
  prob = prob * prob
  classes = scores.shape[1]
  import numpy as np
  threshold = np.repeat(0.5, classes)
  for c in range(classes):
    if c == unknown_ind:
      continue
    n = torch.sum(torch.where(y_true == c, torch.ones_like(y_true), torch.zeros_like(y_true)))
    s = torch.sum(torch.where(y_true == c, prob, torch.zeros_like(y_true)))
    std = (s / n).item()
    threshold[c] = max(0.5, 1 - 3 * std)
  
  if int(os.environ['gpus']) == 0:
    device = 'cpu'
  elif int(os.environ['gpus']) == 1:
    device = os.environ['device']
  else:
    device = os.environ['device'][:6]
  threshold = threshold.to(device=device)
  return threshold