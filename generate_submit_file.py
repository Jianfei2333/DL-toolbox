# Global environment setup.
import os
# Arg parser
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run()
globalconfig.update_parser_params(args)

os.environ['savepath'] += args['model'] + '/'
ind = os.environ['datapath'][:-1].rfind('/')+1
os.environ['datapath'] = os.environ['datapath'][:ind] + args['data'] + '/'

print ('Testing {} with {}.(Running on {})'.format(os.environ['savepath'], os.environ['datapath'], os.environ['device']))

from efficientnet_pytorch import EfficientNet

from tools import datainfo
import torchvision.transforms as T

info = datainfo.getdatainfo()
mean = info['mean']
std = info['std']
normalize = T.Normalize(mean=mean, std=std)

from DataUtils import isic2018_val as data

transform = {
  'val': T.Compose([
    T.Resize((300,300)),
    T.ToTensor(),
    normalize
  ])
}

loader = data.getdata(transform)

import torch.nn as nn
import numpy as np
import pandas as pd

# DEFINE MODEL
models = [None, None, None, None, None]
for i in range(5):
  models[i] = EfficientNet.from_pretrained('efficientnet-b3')
  # Modify.
  num_fcin = models[i]._fc.in_features
  models[i]._fc = nn.Linear(num_fcin, len(info['classes']))

# print (model)

if args['continue']:
  models = globalconfig.loadmodels(models)
else:
  for i in range(5):
    models[i].step=0
    models[i].epochs=0

for i in range(5):
  models[i] = models[i].to(device=os.environ['device'])

from tools import train_and_check as mtool

files = np.array(loader.dataset.imgs)
files = [x[x.rfind('/')+1:x.rfind('.')] for x in files[:,0]]

print(files)

mean_scores = None
for i in range(5):
  scores = mtool.getScores(loader, models[i])
  print (scores.shape)
  if mean_scores is None:
    mean_scores = scores/5
  else:
    mean_scores += scores/5

exp = np.exp(mean_scores)
s = np.sum(exp, axis=1)[:,None]
mean_probability = exp/s

result_mat = np.hstack((np.array(files)[:,None], mean_probability))
head = np.append(['image'], info['classes'])
result_df = pd.DataFrame(result_mat, columns=head)
print (result_df)

result_df.to_csv('{}{}_submit.csv'.format(os.environ['savepath'], args['data']), index=False)