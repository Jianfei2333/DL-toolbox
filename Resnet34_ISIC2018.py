# Global environment setup.
import os
# Arg parser
from config import globalparser
parser = globalparser.getparser()
args = parser.parse_args()

from config import globalconfig
globalconfig.run()
globalconfig.update_filename(__file__)
globalconfig.update_parser_params(args)
args = vars(args)

# Essential network building blocks.
from Networks import Resnet

# Data loader.
# from DataUtils import cifar10
from DataUtils import isic2018_new as isic2018

# Official packages.
import torch
import torch.nn as nn
import torch.optim as optim

# # Training setup.
# TRAIN_EPOCHS=50
# LEARNING_RATE=1e-6

# # 设置从头训练/继续训练
# continue_train=False
# PRETRAIN_EPOCHS=0

# import sys
# args = sys.argv[1:]
# for arg in args:
#   if arg.find('--learning-rate=') != -1:
#     LEARNING_RATE=float(arg[16:])
#     continue
#   if arg.find('--batch-size=') != -1:
#     os.environ['batch-size'] = arg[13:]
#     continue
#   if arg.find('--print-every=') != -1:
#     os.environ['print_every='] = arg[14:]
#     continue
#   if arg.find('--save-every=') != -1:
#     os.environ['save_every'] = arg[13:]
#     continue
#   if arg.find('--epochs=') != -1:
#     TRAIN_EPOCHS = int(arg[9:])
#     continue
#   if arg.find('--continue') != -1:
#     continue_train = True
#     continue
#   if arg.find('--pretrain=') != -1:
#     PRETRAIN_EPOCHS = int(arg[11:])
#     continue
#   else:
#     print('Args error!', arg, 'not found!')
#     sys.exit()

# step=0

# 下面开始进行主干内容

# GOT DATA
train_dataloader, val_dataloader, weights = isic2018.getdata()

# DEFINE MODEL
model = Resnet.Resnet34()

# if continue_train:
#   model_checkpoint = torch.load(pretrain_model_path)
#   model.load_state_dict(model_checkpoint['state_dict'])
#   print('Checkpoint restored!')
#   step = model_checkpoint['episodes']
#   os.environ['tb-logdir'] = model_checkpoint['tb-logdir']

if args['continue']:
  model = globalconfig.loadmodel(model)

# DEFINE OPTIMIZER
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

# Useful tools.
from tools import train_and_check as mtool

# # RUN TRAINING PROCEDURE
mtool.train(
  model,
  optimizer,
  train_dataloader,
  val_dataloader,
  args['epochs']
)
