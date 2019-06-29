import os
import torch.nn as nn

def load(model):
  d = int(os.environ['gpus'])
  if d == 0:
    model = model.to(device='cpu')
  elif d == 1:
    model = model.to(device=os.environ['device'])
  else:
    ids = os.environ['device'][5:]
    ids = [int(x) for x in ids.split(',')]
    model = nn.DataParallel(model, device_ids=ids)
    model = model.to(device='cuda:{}'.format(ids[0]))
  return model