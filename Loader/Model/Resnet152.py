from torchvision import models as Models
import torch.nn as nn

from config import globalconfig
import os

def load(info, Continue=False):
  models = [None, None, None, None, None]
  for i in range(5):
    models[i] = Models.resnet152(pretrained=True)
    # Modify.
    num_fcin = models[i].fc.in_features
    models[i].fc = nn.Linear(num_fcin, len(info['classes']))

  # print (model)

  if Continue:
    models = globalconfig.loadmodels(models)
  else:
    for i in range(5):
      models[i].step=0
      models[i].epochs=0

  params = []
  for i in range(5):
    models[i] = models[i].to(device=os.environ['device'])
    params_to_update = []
    for name,param in models[i].named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
    params.append(params_to_update)
  
  modelinfo = {
    'inputsize': (224, 224)
  }

  return (models, params, modelinfo)