import os
user = os.popen('whoami').readline()

import config.device_selector as d

if user.find('jianfei') == -1:
  os.environ['where_am_i'] = 'pc'
  os.environ['datapath'] = ''
  os.environ['device'] = 'cpu'
else:
  os.environ['where_am_i'] = 'lab'
  os.environ['datapath'] = ''
  os.environ['device'] = 'cuda:'+d.get_gpu_choice()