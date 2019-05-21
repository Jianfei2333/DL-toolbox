def flatten(x):
  """
  Flatten data.
  Resize the data with C channels and resolution of H*W into a vector

  Args:
    x: Tensor with shape (N, C, H, W)
  
  Returns:
    A tensor with shape (N, C*H*W)
  """
  N = x.shape[0] # read in N, C, H, W
  return x.view(N, -1)
