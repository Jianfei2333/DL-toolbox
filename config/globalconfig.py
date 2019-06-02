import os
import sys
import time
from torch import load

def run():
  """
  Run global configuration.
  
  Args:
    - None.

  Return:
    None, but set environment parameters.
      - where_am_i: 'pc' or 'lab', showing where this project is running.
      - datapath: Data dir.kk
      - device: Device configuration with auto recommendation, namely, 'cpu' or 'cuda:#id'
      - tb-logdir: Tensorboard log dir.
  """
  user = os.popen('whoami').readline()
  from tools import deviceSelector as d
  os.environ['step'] = '0'
  if user.find('jianfei') == -1:
    os.environ['where_am_i'] = 'pc'
    os.environ['datapath'] = '/home/huihui/Data/ISIC2019/'
    os.environ['device'] = 'cpu'
    os.environ['tb-logdir'] = '/home/huihui/Log/tensorboard-log/'
    os.environ['logfile-dir'] = '/home/huihui/Log/runlog/'
    os.environ['savepath'] = '/home/huihui/Models/'
  else:
    os.environ['where_am_i'] = 'lab'
    os.environ['datapath'] = '/data0/share/ISIC2019/'
    os.environ['device'] = 'cuda:'+d.get_gpu_choice()
    os.environ['tb-logdir'] = '/data0/jianfei/tensorboard-log/'
    os.environ['logfile-dir'] = '/data0/jianfei/runlog/'
    os.environ['savepath'] = '/data0/jianfei/models/'
  t = time.asctime().replace(' ', '-')
  os.environ['logfile-dir'] += t
  os.environ['tb-logdir'] += t

  print('Finish global configuration!')

def update_filename(file):
  """
  Update logdir to take filename.

  Args:
    file: String of __file__, which is the full path of file.

  Return:
    None, but set environment parameters.
      tb-logdir: tb-logdir + '-' + filename
      filename: String of entrance file name.(Without postfix '.py')
      savepath: savepath + filename + '/'
  """
  s = file.rfind('/')
  e = file.find('.')
  filename = file[s+1:e]
  os.environ['tb-logdir'] += '-' + filename
  os.environ['savepath'] += filename + '/'
  os.environ['logfile-dir'] += filename + '.log'
  os.environ['filename'] = filename

def update_parser_params(args):
  """
  """
  os.environ['batch-size'] = args['batch_size']
  os.environ['print_every'] = args['print_every']
  os.environ['save_every'] = args['save_every']
  os.environ['pretrain-modelpath'] = os.environ['savepath']+args['pretrain']+'epochs.pkl'
  os.environ['pretrain-epochs'] = args['pretrain']

def loadmodel(model):
  checkpoint = load(os.environ['pretrain-modelpath'])
  model.load_state_dict(checkpoint['state_dict'])
  print('Checkpoint restored!')
  os.environ['step'] = checkpoint['episodes']
  os.environ['tb-logdir'] = checkpoint['tb-logdir']
  return model