import argparse

def getparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-lr', '--learning-rate', type=float, help='Assign learning rate, default 1e-4.', default=1e-4)
  parser.add_argument('-b', '--batch-size', help='Assign batch size, default 64.', default='64')
  parser.add_argument('-c', '--continue', help='Continue training, default False.', default=False, action='store_true')
  parser.add_argument('-e', '--epochs', type=int, help='Epochs to train, default 20.', default=20)
  parser.add_argument('--save-every', help='Save the model every n epochs, default 5.', default='5')
  parser.add_argument('--print-every', help='Print the model every n steps, default 10.', default='50')
  parser.add_argument('--gpus', help='Choose the total number of gpus to use.', default=1)
  parser.add_argument('--data', help='Set the data path.')
  parser.add_argument('--model', help='Set the model path.')
  return parser
