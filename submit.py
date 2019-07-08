# Global environment setup.
import os
# Arg parser
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run(args, False)

print ('Generating submit file {} with {}.(Running on {})'.format(os.environ['savepath'], os.environ['datapath'], os.environ['device']))

import importlib
# Essential network building blocks.
model = importlib.import_module('Loader.Model.'+args['model'])
transform = importlib.import_module('Loader.Transform.basic')

from DataUtils import ImgFolder_val as data

import numpy as np
import pandas as pd

# Create models and dataloaders
from tools import datainfo
info = datainfo.getdatainfo(os.environ['datapath'])

models, params, modelinfo = model.load(info, True)

transform = transform.load(modelinfo, info)

loader = data.getdata(args['type'], transform)

from tools import trainer as mtool

# Get Filenames
files = np.array(loader.dataset.imgs)
files = [x[x.rfind('/')+1:x.rfind('.')] for x in files[:,0]]
# print(files)

# Print scores and probabilities
mean_scores = None
for i in range(5):
  scores = mtool.getScores(loader, models[i])
  print ("{}: shape {}".format(i, scores.shape))
  if mean_scores is None:
    mean_scores = scores
  else:
    mean_scores += scores

exp = np.exp(mean_scores)
s = np.sum(exp, axis=1)[:,None]
mean_probability = exp/s

# Generate submit file
result_mat = np.hstack((np.array(files)[:,None], mean_probability))
head = np.append(['image'], info['classes'])
result_df = pd.DataFrame(result_mat, columns=head)
print (result_df)

result_df.to_csv('{}{}_submit.csv'.format(os.environ['savepath'], args['data']), index=False)