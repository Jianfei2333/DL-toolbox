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
  if os.environ['damp'] == 'cosine':
    damper = {
      0: lambda x: x/5,
      1: lambda x: x*2,
      2: lambda x: x*(3/2),
      3: lambda x: x*(4/3),
      4: lambda x: x*(5/4),
    }
    import math
    for i in range(6, 200):
      damper[i] = lambda x: x * ((1+math.cos((i-5)*math.pi/195))/(1+math.cos((i-6)*math.pi/195)))
    if epoch in damper.keys():
      for g in optimizer.param_groups:
        g['lr'] = damper[epoch](g['lr'])
    return optimizer