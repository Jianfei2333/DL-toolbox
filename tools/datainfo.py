import json

def getdatainfo(datapath):
  filename = datapath+'info.json'
  f = open(filename)
  data = json.load(f)
  return data