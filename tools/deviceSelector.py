"""
Device selector.
A useful script to read the GPU status
  and choose the best one automatically.
"""
import os

def parse(line):
  """
  Parse a nvidia-smi query line into parameters.

  Args:
    - line: A csv-like string.
  
  Return:
    - params: A list of parameters.(Type string)
  """
  params = []
  while line.find(',') != -1:
    params.append(line[:line.find(',')].strip())
    line = line[line.find(',')+1:]
  params.append(line.strip())
  return params

def get_gpus():
  """
  Get gpu status.

  Args:
    - None.
  
  Return:
    - gpus: A dictionary with gpu status, parameters as follows.
      - 'index': GPU id.
      - 'memfree': Free memory of this GPU.(MiB)
      - 'memtotal': Total memory of this GPU.(MiB)
      - 'usage': GPU usage.(Percent)
  """
  gpulist = os.popen('nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader').readlines()
  gpus = []
  for val in gpulist:
    params = parse(val)
    gpu = {}
    gpu['index'] = int(params[0])
    gpu['memfree'] = int(params[1][:params[1].find('MiB')].strip())
    gpu['memtotal'] = int(params[2][:params[2].find('MiB')].strip())
    gpu['usage'] = int(params[3][:params[2].find('%')].strip())
    gpus.append(gpu)
  return gpus

def get_gpu_choice():
  """
  Get the best GPU choice.
  
  Args:
    - None.
  
  Return:
    - device: A string variable representing the GPU choice.
        For example: 'cuda:0'.
  """
  gpus = get_gpus()
  r = sorted(gpus, key=lambda v: (((float)(v['memtotal']-v['memfree']))/v['memtotal'])*100+v['usage'])[0]
  if ((float)(r['memfree'])/r['memtotal']) < 0.5:
    print('###############################################################################')
    print('Warning: The best gpu now is in high load! Please check and use another server!')
    print('###############################################################################')
  print ('Recommended gpu is: gpu%d, mem: %d/%d (%.2f%%), usage: %d%%' % (r['index'], r['memfree'], r['memtotal'], ((float)(r['memfree'])/r['memtotal'])*100, r['usage']))
  return str(r['index'])