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
from DataUtils import ImgFolder_5fold_openset as data
from DataUtils import ImgFolder as data_reference

# Official packages.
import torch.nn as nn
import torch.optim as optim

# 下面开始进行主干内容
from tools import datainfo
info = datainfo.getdatainfo(os.environ['datapath'])
info_reference = datainfo.getdatainfo(os.environ['datapath'].replace(args['data'], 'support'))

models, params, modelinfo1 = model.load(info, args['continue'])
model1 = models[0]
params1 = params[0]

models, params, modelinfo2 = model.load(info_reference, args['continue'])
model2 = models[0]
params2 = params[0]

transform1 = transform.load(modelinfo1, info)
transform2 = transform.load(modelinfo2, info_reference)

# GOT DATA
dataloader1 = data.getdata(transform1)[0]
dataloader2 = data_reference.getdata(transform2)[0]

print(dataloader1['train'].dataset.__len__())
print(dataloader2['train'].dataset.__len__())

# DEFINE OPTIMIZER
optimizer1 = optim.SGD(params1, lr=args['learning_rate'], momentum=0.9)
optimizer2 = optim.SGD(params2, lr=args['learning_rate'], momentum=0.9)

criterion = importlib.import_module('Loader.Loss.Membership').loss

mtool = importlib.import_module('tools.trainer-DeepTrans')

mtool.train(
  model1,
  model2,
  dataloader1,
  dataloader2,
  optimizer1,
  optimizer2,
  criterion,
  args['epochs']
)
