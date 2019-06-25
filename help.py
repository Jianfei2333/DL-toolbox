# Global environment setup.
import os
# Arg parser
from config import globalparser
args = vars(globalparser.getparser().parse_args())

from config import globalconfig
globalconfig.run(args, False)

import glob
import numpy as np

def print_prompt():
  available_data = np.array([])
  datalist = glob.glob(os.environ['datapath']+'/*')
  for p in datalist:
    if os.path.exists(p+'/info.json'):
      available_data = np.append(available_data, p)
  available_data = [x[x.rfind('/')+1:] for x in available_data]
  print ('Available datas:')
  print (available_data)

  modellist = glob.glob('./Loader/Model/*.py')
  modellist = [x[x.rfind('/')+1:x.rfind('.py')] for x in modellist]

  print ('Available models:')
  print (modellist)
  

print_prompt()