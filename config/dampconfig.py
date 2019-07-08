import numpy as np
import os

def damp(optimizer, epoch):
  if os.environ['damp'] == '0':
    return optimizer
  if os.environ['damp'] == '1':
    damper = {
      40: lambda x: x/3,
      60: lambda x: x*0.3,
      80: lambda x: x/3,
      90: lambda x: x*0.3,
      100: lambda x: x/3,
      110: lambda x: x*0.3,
      120: lambda x: x/3,
      130: lambda x: x*0.3,
      140: lambda x: x/3,
      150: lambda x: x*0.3,
      160: lambda x: x/3
    }
    if epoch in damper.keys():
      for g in optimizer.param_groups:
        g['lr'] = damper[epoch](g['lr'])
    return optimizer