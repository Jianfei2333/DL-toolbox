import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

def writer():
  """
  The global tensorboard writer configuration and getter.
  
  Args:
    None.

  Return:
    writer: Tensorboard SummaryWriter with configuration.
  """
  writer = SummaryWriter(log_dir=os.environ['logdir'])
  return writer