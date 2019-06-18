import os
import json

def getdatainfo():
  filename = os.environ['datapath']+'info.json'
  f = open(filename)
  data = json.load(f)
  return data