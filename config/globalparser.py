import argparse

def getparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-lr', '--learning-rate', type=float, help='Assign learning rate, default 3e-2.', default=3e-2)
  parser.add_argument('-b', '--batch-size', help='Assign batch size, default 64.', default='64')
  parser.add_argument('--batch-scale', help='Assign batch scale, the model will update its params every k batches. The actual batch size will be k*b', default='1')
  parser.add_argument('-c', '--continue', help='Continue training, default False.', default=False, action='store_true')
  parser.add_argument('-e', '--epochs', type=int, help='Epochs to train, default 10.', default=10)
  parser.add_argument('--save-every', help='Save the model every n epochs, default 5.', default='5')
  parser.add_argument('--print-every', help='Print the model every n steps, default 300.', default='300')
  parser.add_argument('--data', help='Set the data path.', default='')
  parser.add_argument('--model', help='Set the model path.', default='')
  parser.add_argument('--loss', help='Set the loss function.', default='')
  parser.add_argument('--remark', help='Add a remark after every path. Default `debug`.', default='debug')
  parser.add_argument('--damp', help='Add a learning-rate damp, which was predefined. Default 1.', default='1')
  parser.add_argument('--gpus', type=int, help='Choose the total number of gpus to use.', default=1)
  parser.add_argument('--type', help='Type of submit file to generate, default "Val", choice ("Val", "Test"). Used in generating submission file.', default='Val')
  return parser
