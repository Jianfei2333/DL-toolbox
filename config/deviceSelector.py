import os

def parse(line):
  params = []
  while line.find(',') != -1:
    params.append(line[:line.find(',')].strip())
    line = line[line.find(',')+1:]
  params.append(line.strip())
  return params

def get_gpus():
  gpulist = os.popen('nvidia-smi --query-gpu=index,utilization.memory,utilization.gpu --format=csv,noheader').readlines()
  gpus = []
  for val in gpulist:
    params = parse(val)
    gpu = {}
    gpu['index'] = int(params[0])
    gpu['mem'] = int(params[1][:params[1].find('%')].strip())
    gpu['usage'] = int(params[2][:params[2].find('%')].strip())
    gpus.append(gpu)
  return gpus

def get_gpu_choice():
  gpus = get_gpus()
  r = sorted(gpus, key=lambda v: v['mem']*100+v['usage'])[0]
  if r['mem'] >= 50:
    print('###############################################################################')
    print('Warning: The best gpu now is in high load! Please check and use another server!')
    print('###############################################################################')
  print ('Recommended gpu is: gpu%d, mem: %d%%, usage: %d%%' % (r['index'], r['mem'], r['usage']))
  return r['index']