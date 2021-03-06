# Global environment setup.
import os
# Arg parser
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run(args)

print ('Train {} with {}.(Running on {})'.format(os.environ['savepath'], os.environ['datapath'], os.environ['device']))

import importlib
# Essential network building blocks.
model = importlib.import_module('Loader.Model.'+args['model'])
transform = importlib.import_module('Loader.Transform.train_aug')

# Data loader.
from DataUtils import ImgFolder_5fold as data

# Official packages.
import torch.nn as nn
import torch.optim as optim

# 下面开始进行主干内容
from tools import datainfo
info = datainfo.getdatainfo(os.environ['datapath'])

models, params, modelinfo = model.load(info, args['continue'])

transform = transform.load(modelinfo, info)

# GOT DATA
dataloaders = data.getdata(transform)

# DEFINE MODEL

# DEFINE OPTIMIZER
optimizers = [None, None, None, None, None]
for i in range(5):
  optimizers[i] = optim.SGD(params[i], lr=args['learning_rate'], momentum=0.9)
  # optimizer = optim.Adam(params[i], lr=args['learning_rate'])

criterion = nn.functional.cross_entropy
# criterion = RecallWeightedCrossEntropy.recall_cross_entropy

from tools import train_and_check as mtool

mtool.train5folds(
  models,
  dataloaders,
  optimizers,
  criterion,
  args['epochs']
)