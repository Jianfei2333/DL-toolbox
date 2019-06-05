import torch
import os
import numpy as np
import pandas as pd
import time
from config.globaltb import writer
from tools.metrics import cmatrix, precision_recall, accuracy
import sklearn.metrics as metrics
import copy

writer = writer()

def check(loader, model, step=0):
  """
  Check the accuracy of the model on validation set / test set.

  Args:
    loader: A Torchvision DataLoader with data to check.
    model: A PyTorch Module giving the model to check.

  Return:
    Nothing, but print the accuracy result to console.
  """
  print ('Checking accuracy on validation set.\n')

  device=os.environ['device']

  classes = loader.dataset.classes
  C = len(classes)
  
  model.eval()
  with torch.no_grad():
    y_pred = None
    y_true = None
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)
      _, preds = scores.max(1)
      # Prediction array
      if y_pred is None:
        y_pred = preds.cpu().numpy()
      else:
        y_pred = np.hstack((y_pred, preds.cpu().numpy()))
      # Groundtruth array
      if y_true is None:
        y_true = y.cpu().numpy()
      else:
        y_true = np.hstack((y_true, y.cpu().numpy()))

    met_acc = accuracy(y_true, y_pred)
    met_confusion_matrix = cmatrix(y_true, y_pred, classes)
    met_precision_recall = precision_recall(y_true, y_pred, classes)
    met_balanced_acc_score = metrics.balanced_accuracy_score(y_true, y_pred)

    # Print result
    print ('Acc:\t%.4f' % met_acc)
    print()
    print ('Balanced accuracy score:\t%.4f' % met_balanced_acc_score)
    print()
    print ('Confusion matrix:\t')
    print (met_confusion_matrix)
    print()
    print ('Precision and Recall:\t')
    print (met_precision_recall)
    print()

    writer.add_scalars('Aggregate/Acc',{'Validation Acc': met_acc}, step)
    for i in range(C):
      writer.add_scalars('Multiclass/'+classes[i], {
        'Precision': met_precision_recall[classes[i]][0],
        'Recall': met_precision_recall[classes[i]][1]
      }, step)
    writer.add_scalars('Aggregate/BalancedAcc', {'Score': met_balanced_acc_score}, step)
    
    return met_balanced_acc_score

def train(
  model,
  dataloader,
  optimizer,
  criterion,
  epochs=1
):
  """
  Train a model with optimizer using PyTorch API.

  Args:
    model: A PyTorch Module giving the model to train.
    optimizer: An Optimizer object used to train the model.
    train_dataloader: A Torchvision DataLoader with training data.
    val_dataloader: A Torchvision DataLoader with validation data.
    test_dataloader: A Torchvision DataLoader with test data.
    pretrain_epochs: (Optional) A Python integet giving the number of epochs the model pretrained.
    epochs: (Optional) A Python integer giving the number of epochs to train for.
    step: (Optional) A Python integer giving the number of steps the model pretrained.

  Returns:
    Nothing, but prints model accuracies during training.
  """
  since = time.time()
  train_dataloader = dataloader['train']
  val_dataloader = dataloader['val']

  step = int(os.environ['step'])
  pretrain_epochs = int(os.environ['pretrain-epochs'])
  device = os.environ['device']
  model = model.to(device=device)

  # Weights: The number of samples in each class.
  # Train_weights: 1/Weights, with normalization.
  weights = train_dataloader.dataset.weights
  train_weights = 1 / weights
  s = np.sum(train_weights)
  train_weights = torch.from_numpy(train_weights / s).to(device=device, dtype=torch.float32)

  # Print every n steps.
  print_every = int(os.environ['print_every'])
  # Save model every n epochs.
  save_every = int(os.environ['save_every'])

  best_model = copy.deepcopy(model.state_dict())
  best_balance_acc = 0.0

  for e in range(epochs):
    running_y = np.array([])
    running_ypred = np.array([])
    for t, (x, y) in enumerate(train_dataloader):
      model.train()
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      # Forward prop.
      scores = model(x)
      _, preds = scores.max(1)
      preds = preds.cpu().numpy()
      running_ypred = np.hstack((running_ypred, preds))
      running_y = np.hstack((running_y, y.cpu().numpy()))
      loss = criterion(scores, y, train_weights)

      writer.add_scalars('Aggregate/Loss',{'loss': loss.item()}, step)
      step += 1

      # Back prop.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      
      if (t+1) % print_every == 0:
        met_acc = accuracy(running_y, running_ypred)
        met_balanced_acc_score = metrics.balanced_accuracy_score(running_y, running_ypred)
        writer.add_scalars('Aggregate/Acc',{'Train Acc': met_acc}, step)
        writer.add_scalars('Aggregate/BalancedAcc', {'Train Score': met_balanced_acc_score}, step)
        print ('* * * * * * * * * * * * * * * * * * * * * * * *')
        sec = time.time() - since
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int((sec % 3600) % 60)
        elapse = "{} hours, {} minutes, {} seconds.".format(h,m,s)
        print (time.asctime().replace(' ', '-'), ' Elapsed time:', elapse)
        print('Epoch %d/%d, Step %d (Total %d/%d, %d):\nLoss:\t%.4f\nTraining acc\t%.4f\nTraining balanced score\t%.4f' % (e+1, epochs, t+1, e+1+pretrain_epochs, epochs+pretrain_epochs, step, loss.item(), met_acc, met_balanced_acc_score))
        print ('* * * * * * * * * * * * * * * * * * * * * * * *')
        res = check(val_dataloader, model, step)
        print()
        if res > best_balance_acc:
          best_model = copy.deepcopy(model.state_dict())
          best_balance_acc = res

    if (e+1) % save_every == 0:
      savepath = os.environ['savepath']
      savefilepath = savepath + str(e+pretrain_epochs+1) + 'epochs.pkl'
      # Check if the savepath is valid.
      if not os.path.exists(savepath):
        os.mkdir(savepath)
        print ('Create dir', savepath)
      # Save the checkpoint.
      torch.save({
        'state_dict': model.state_dict(),
        'episodes': str(step),
        'tb-logdir': os.environ['tb-logdir']
        },
        savefilepath
      )
      print ('Model save as', savefilepath)

  torch.save({
      'state_dict': best_model,
      'episodes': str(step),
      'tb-logdir': os.environ['tb-logdir'],
      'epochs': str(e+pretrain_epochs+1)
    },
    os.environ['savepath'] + 'best.pkl'
  )