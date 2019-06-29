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
  if create and not os.path.exists(os.environ['savepath']):
    os.mkdir(os.environ['savepath'])
    for i in range(5):
      os.mkdir(os.environ['savepath']+'fold{}/'.format(i))
    print ('Create dir', os.environ['savepath'])

  os.environ['batch-size'] = args['batch_size']
  os.environ['print_every'] = args['print_every']
  os.environ['save_every'] = args['save_every']
  os.environ['gpus'] = str(args['gpus'])
  
  if os.environ['where_am_i'] == 'pc' or args['gpus'] == 0:
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

def loadmodels(models):
  for i in range(5):
    filepath = '{}fold{}/best.pkl'.format(os.environ['savepath'], i)
    if (os.path.exists('{}fold{}/best.pkl'.format(os.environ['savepath'], i))):
      checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
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
