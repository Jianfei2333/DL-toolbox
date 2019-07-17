import torch
import os
import numpy as np

def prediction(input, unknown_ind, t = 0.5, weight=None):
  softmax = torch.exp(input) / torch.exp(input).sum(dim=1)[:, None]
  _, pred1 = softmax.max(1)
  info = -softmax * torch.log(softmax)
  info = torch.cat([info[:, :unknown_ind], info[:, unknown_ind+1:]], dim=1)
  entropy = torch.sum(info, dim=1)
  
  a = torch.from_numpy(np.repeat(1/info.shape[1], info.shape[1])).type(torch.float)
  top = torch.sum(-a * torch.log(a)).item()
  entropy = entropy / top

  if int(os.environ['gpus']) == 0:
    device = 'cpu'
  elif int(os.environ['gpus']) == 1:
    device = os.environ['device']
  else:
    device = os.environ['device'][:6]
  pred = torch.where(entropy < t, pred1, torch.tensor([unknown_ind]).to(device=device))

  return pred