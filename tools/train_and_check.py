import torch
import os
from Utils.globaltb import writer

writer = writer()

def checkAcc(loader, model, step=0):
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
    print('##############################')
    print('Checking accuracy on test set.')
    print('##############################')

  device=os.environ['device']

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
    prompt = 'Got %d / %d correct: %.2f%%' % (num_correct, num_samples, 100 * acc)
    print(prompt)
    if loader.dataset.train:
      writer.add_scalars('Train/Acc',{'Acc': acc}, step)
    else:
      writer.add_text(os.environ['filename'], prompt + 'on TEST set.', step)

def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, epochs=1):
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
  device = os.environ['device']

  model = model.to(device=device)
  step = 1
  for e in range(epochs):
    for t, (x, y) in enumerate(train_dataloader):
      model.train()
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      scores = model(x)
      loss = torch.nn.functional.cross_entropy(scores, y)

      writer.add_scalars('Train/loss',{'loss': loss.item()}, step)
      step += 1

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()
      

      if t % int(os.environ['print_every']) == 0:
        print('Iteration %d, loss = %.4f' % (t, loss.item()))
        checkAcc(val_dataloader, model, step)
        print()

    if e % int(os.environ['save_every']) == 0:
      checkAcc(test_dataloader, model, e)
      savepath = os.environ['savepath'] + os.environ['filename'] + '/'
      if not os.path.exists(savepath):
        os.mkdir(savepath)
        print ('Create dir', savepath)
      torch.save(model.state_dict(), savepath + str(e+1) + 'epochs.pkl')
      print ('Model save as', savepath + str(e+1) + 'epochs.pkl')