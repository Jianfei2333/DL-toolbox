import os

# Abondon. 2019-05-23
def modelvisual(model, writer, data):
  """
  Generate model graph in tensorboard.

  Args:
    model: A PyTorch model to generate.
    writer: A Tensorboard SummaryWriter with configuration.
    data: A torchvision dataloader with training data.
  """
  d, _ = next(iter(data))
  writer.add_graph(model, d)