def flatten(x):
  '''
  Flatten data
  input: x, shape in N, C, H, W
  output: matrix shape N, C*H*W
  '''
  N = x.shape[0] # read in N, C, H, W
  return x.view(N, -1)