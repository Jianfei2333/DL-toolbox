import torch

def checkAcc(loader, model):
  """
  Check the accuracy of the model on validation set / test set.

  Args:
    loader: A Torchvision DataLoader with data to check.
    model: A PyTorch Module giving the model to check.

  Return:
    Nothing, but print the accuracy result to console.
  """
  if loader.dataset.train:
    print('Checking accuracy on validation set.')
  else:
    print('Checking accuracy on test set.')

  if torch.cuda.is_available(): 
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  num_correct = 0
  num_samples = 0
  model.eval()
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct: %.2f%%' % (num_correct, num_samples, 100 * acc))

def train(model, optimizer, train_dataloader, val_dataloader, epochs=1):
  """
  Train a model with optimizer using PyTorch API.

  Args:
    model: A PyTorch Module giving the model to train.
    optimizer: An Optimizer object used to train the model.
    train_dataloader: A Torchvision DataLoader with training data.
    val_dataloader: A Torchvision DataLoader with validation data.
    epochs: (Optional) A Python integer giving the number of epochs to train for.

  Returns:
    Nothing, but prints model accuracies during training.
  """
  if torch.cuda.is_available():
    print ('Training on device: gpu')
    device = torch.device('cuda')
  else:
    print ('Training on device: cpu')
    device = torch.device('cpu')

  model = model.to(device=device)
  for e in range(epochs):
    for t, (x, y) in enumerate(train_dataloader):
      model.train()
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      scores = model(x)
      loss = torch.nn.functional.cross_entropy(scores, y)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      if t % 100 == 0:
        print('Iteration %d, loss = %.4f' % (t, loss.item()))
        checkAcc(val_dataloader, model)
        print()
