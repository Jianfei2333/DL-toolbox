import numpy as np
import torchvision.transforms as T

def load(modelinfo, info):
  mean = info['mean']
  std = info['std']
  normalize = T.Normalize(mean=mean, std=std)

  inputsize = modelinfo['inputsize']
  resize = tuple([int(x*(4/3)) for x in inputsize])

  transform = {
    'val': T.Compose([
      T.Resize(resize),
      T.ToTensor(),
      normalize
    ])
  }
  
  return transform