import os
import torch.nn as nn
import torch

def load(model):
  d = int(os.environ['gpus'])
  if d == 0:
    model_p = model.to(device='cpu')
  elif d == 1:
    id = os.environ['device'][5:]
    id = int(id)
    model_p = nn.DataParallel(model, device_ids=[id])
    model_p = model_p.to(device=os.environ['device'])
  else:
    ids = os.environ['device'][5:]
    ids = [int(x) for x in ids.split(',')]
    model_p = nn.DataParallel(model, device_ids=ids)
    if hasattr(model, 'step'):
      step = model.step
      model_p.step = step
    if hasattr(model, 'epochs'):
      epochs = model.epochs
      model_p.epochs = epochs
    model_p = model_p.to(device='cuda:{}'.format(ids[0]))
    # model_p = model_p.to(device='cuda:{}'.format(ids[0]))
  return model_p

def move(model):
  d = int(os.environ['gpus'])
  if d >= 1:
    ids = os.environ['device'][5:]
    ids = [int(x) for x in ids.split(',')]
    model = model.to(device='cuda:{}'.format(ids[0]))
  return model

def moveback(model):
  d = int(os.environ['gpus'])
  if d >= 1:
    ids = os.environ['device'][5:]
    ids = [int(x) for x in ids.split(',')]
    model = model.to(device='cpu')
    torch.cuda.set_device('cuda:{}'.format(ids[0]))
    torch.cuda.empty_cache()
  return model
