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
model = models[0]
params = params[0]

transform = transform.load(modelinfo, info)

# GOT DATA
dataloader = data.getdata(transform)[0]

# DEFINE MODEL

# DEFINE OPTIMIZER
optimizer = optim.SGD(params, lr=args['learning_rate'], momentum=0.9)

criterion = importlib.import_module('Loader.Loss.DOC').loss

mtool = importlib.import_module('tools.trainer-DOC')

mtool.train(
  model,
  dataloader,
  optimizer,
  criterion,
  args['epochs']
)