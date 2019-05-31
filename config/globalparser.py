import argparse

def getparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-lr', '--learning-rate', type=float, help='Assign learning rate, default 1e-5.', default=1e-5)
  parser.add_argument('-b', '--batch-size', help='Assign batch size, default 64.', default='64')
  parser.add_argument('-c', '--continue', help='Continue training, default False.', default=False, action='store_true')
  parser.add_argument('-p', '--pretrain', help='Pretrain epochs.', default='0')
  parser.add_argument('-e', '--epochs', help='Epochs to train, default 20.', default=20)
  parser.add_argument('--save-every', help='Save the model every n epochs, default 10.', default='10')
  parser.add_argument('--print-every', help='Print the model every n steps, default 10.', default='10')
  return parser
