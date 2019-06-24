import os
import sys
import time
from torch import load
import torch

def run(args, create=True):
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
  model = args['model']
  data = args['data']
  # Global environment variables.
  user = os.popen('whoami').readline()
  from tools import deviceSelector as d
  # os.environ['step'] = '0'
  if user.find('jianfei') == -1:
    os.environ['where_am_i'] = 'pc'
    os.environ['datapath'] = '/home/huihui/Data/'
    os.environ['tb-logdir'] = '/home/huihui/Log/tensorboard-log/'
    os.environ['logfile-dir'] = '/home/huihui/Log/runlog/'
    os.environ['savepath'] = '/home/huihui/Models/'
  else:
    os.environ['where_am_i'] = 'lab'
    os.environ['datapath'] = '/data0/share/'
    os.environ['tb-logdir'] = '/data0/jianfei/tensorboard-log/'
    os.environ['logfile-dir'] = '/data0/jianfei/runlog/'
    os.environ['savepath'] = '/data0/jianfei/models/'
  t = time.asctime().replace(' ', '-')
  os.environ['datapath'] += data + '/'
  os.environ['logfile-dir'] += t + model + '-' + data + '/'
  os.environ['tb-logdir'] += t + model + '-' + data + '/'
  os.environ['logfile-dir'] += t + model + '-' + data + '.log'
  os.environ['savepath'] += model + '-' + data + '/'
  if not os.path.exists(os.environ['savepath']):
    os.mkdir(os.environ['savepath'])
    for i in range(5):
      os.mkdir(os.environ['savepath']+'fold{}/'.format(i))
    print ('Create dir', os.environ['savepath'])

  os.environ['batch-size'] = args['batch_size']
  os.environ['print_every'] = args['print_every']
  os.environ['save_every'] = args['save_every']
  
  if create and os.environ['where_am_i'] == 'pc' or args['gpus'] == 0:
    os.environ['device'] = 'cpu'
  else:
    from tools import deviceSelector as d
    gs = d.get_gpu_choice(args['gpus'])
    os.environ['device'] = 'cuda:'+gs

  # Some global restricts.
  if os.environ['device'] != 'cpu':
    torch.cuda.set_device(os.environ['device'])
  torch.set_num_threads(1)

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
  if not os.path.exists(os.environ['savepath']):
    os.mkdir(os.environ['savepath'])
    for i in range(5):
      os.mkdir(os.environ['savepath']+'fold{}/'.format(i))
    print ('Create dir', os.environ['savepath'])

def update_parser_params(args):
  """
  """
  from tools import deviceSelector as d
  os.environ['batch-size'] = args['batch_size']
  os.environ['print_every'] = args['print_every']
  os.environ['save_every'] = args['save_every']
  if args['gpus'] != 1:
    if args['gpus'] == 0:
      os.environ['device'] = 'cpu'
    else:
      gs = d.get_gpu_choice(args['gpus'])
      os.environ['device'] = 'cuda:'+gs

def loadmodels(models):
  for i in range(5):
    filepath = '{}fold{}/best.pkl'.format(os.environ['savepath'], i)
    if (os.path.exists('{}fold{}/best.pkl'.format(os.environ['savepath'], i))):
      checkpoint = load(filepath)
      models[i].load_state_dict(checkpoint['state_dict'])
      os.environ['tb-logdir'] = checkpoint['tb-logdir']
      models[i].step = int(checkpoint['step'])
      models[i].epochs = int(checkpoint['epochs'])
  print('Checkpoint restored!') 
  return models

def set_no_grad(model):
  for param in model.parameters():
    param.requires_grad = False
  return model
