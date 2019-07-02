import torch
import os
import numpy as np
import pandas as pd
import time
from config.globaltb import writer
from tools.metrics import cmatrix, precision_recall, accuracy
import sklearn.metrics as metrics
from tools import modelLoader
import copy

writer = writer()

def release(*argv):
  for v in argv:
    v.cpu()
  ids = os.environ['device'][5:]
  ids = [int(x) for x in ids.split(',')]
  torch.cuda.set_device('cuda:{}'.format(ids[0]))
  torch.cuda.empty_cache()

def savemodel(filename, model):
  savefilepath = '{}fold{}/{}'.format(os.environ['savepath'], model.fold, filename)
  # Save the checkpoint.
  torch.save({
    'state_dict': model.state_dict(),
    'step': str(model.step),
    'tb-logdir': os.environ['tb-logdir']
    },
    savefilepath
  )
  print ('Model saved as', savefilepath)

def getElapse(since):
  sec = time.time() - since
  h = int(sec // 3600)
  m = int((sec % 3600) // 60)
  s = int((sec % 3600) % 60)
  return (h, m, s)

def getScores(loader, model):
  device = os.environ['device']
  if int(os.environ['gpus']) > 1:
    device=device[:device.find(',')]

  model.eval()
  with torch.no_grad():
    total_scores = None
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      scores = model(x)
      score = scores.cpu().numpy()
      if total_scores is not None:
        total_scores = np.append(total_scores, score, axis=0)
      else:
        total_scores = score
  return total_scores


def check(loader, model, step, criterion=None, kwargs={'mode':'val'}):
  """
  Check the accuracy of the model on validation set / test set.

  Args:
    loader: A Torchvision DataLoader with data to check.
    model: A PyTorch Module giving the model to check.

  Return:
    Nothing, but print the accuracy result to console.
  """
  mode = kwargs['mode']
  if mode == 'val':
    print ('Checking accuracy on validation set.\n')
  elif mode == 'train':
    print ('* * * * * * * * * * * * * * * * * * * * * * * *')
    print ('Checking accuracy on training set without data augmentation.')
    print ('* * * * * * * * * * * * * * * * * * * * * * * *')

  device=os.environ['device']
  if int(os.environ['gpus']) > 1:
    device=device[:device.find(',')]

  classes = loader.dataset.classes
  C = len(classes)
  
  model.eval()
  with torch.no_grad():
    y_pred = None
    y_true = None
    running_loss = 0.
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)

      if criterion is not None:
        loss_weights = kwargs['loss_weights']
        running_loss += criterion(scores, y, loss_weights).item()

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

    if mode == 'val':
      writer.add_scalars('fold{}/Aggregate/Acc'.format(model.fold),{'Val Acc': met_acc}, step)
      # for i in range(C):
      #   writer.add_scalars('fold{}/Multiclass/'.format(model.fold)+classes[i], {
      #     'Val Precision': met_precision_recall[classes[i]][0],
      #     'Val Recall': met_precision_reall[classes[i]][1]
      #   }, step)
      if criterion is not None:
        writer.add_scalars('fold{}/Aggregate/Loss'.format(model.fold), {'Val Loss': running_loss}, step)
      writer.add_scalars('fold{}/Aggregate/Score'.format(model.fold), {'Val Score': met_balanced_acc_score}, step)
    elif mode == 'train':
      writer.add_scalars('fold{}/Aggregate/Acc'.format(model.fold),{'Train Acc': met_acc}, step)
      # for i in range(C):
      #   writer.add_scalars('fold{}/Multiclass/'.format(model.fold)+classes[i], {
      #     'Train Precision': met_precision_recall[classes[i]][0],
      #     'Train Recall': met_precision_recall[classes[i]][1]
      #   }, step)
      if criterion is not None:
        writer.add_scalars('fold{}/Aggregate/Loss'.format(model.fold), {'Train Loss': running_loss}, step)
      writer.add_scalars('fold{}/Aggregate/Score'.format(model.fold), {'Train Score': met_balanced_acc_score}, step)

    return met_balanced_acc_score

def train_one_epoch(
  model,
  dataloader,
  optimizer,
  criterion,
  info
):
  since = info['since']
  e = info['e'] + 1
  epochs = info['epochs']
  train_weights = info['train_weights']
  best_score = info['best']['score']
  best_model = info['best']['model']

  device = os.environ['device']
  if int(os.environ['gpus']) > 1:
    device=device[:device.find(',')]
  print_every = int(os.environ['print_every'])
  save_every = int(os.environ['save_every'])
  batch_scale = int(os.environ['batch_scale'])

  train_dataloader = dataloader['train']
  train4val_dataloader = dataloader['train4val']
  val_dataloader = dataloader['val']

  step = model.step
  total_e = model.epochs+e
  total_epochs = model.epochs+epochs

  for t, (x, y) in enumerate(train_dataloader):
    model.train()
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.long)

    # Forward prop.
    scores = model(x)
    loss = criterion(scores, y, train_weights)
    loss = loss/batch_scale

    # writer.add_scalars('fold{}/Aggregate/Loss'.format(model.fold),{'loss': loss.item()}, step)
    step += 1

    # Back prop.
    loss.backward()

    if (t+1) % batch_scale == 0:
      optimizer.step()
      optimizer.zero_grad()

    if (t+1) % print_every == 0:
      print ('* * * * * * * * * * * * * * * * * * * * * * * *')
      h, m, s = getElapse(since)
      elapse = "{} hours, {} minutes, {} seconds.".format(h,m,s)
      print (time.asctime().replace(' ', '-'), ' Elapsed time:', elapse)
      prompt = '''
        Fold {}
        Epoch {}/{}, Step {} (Total {}/{}, {})
      '''.format(
        model.fold,
        e,
        epochs,
        t+1,
        total_e,
        total_epochs,
        step,
        # loss.item()
      )
      print (prompt)
      print ('* * * * * * * * * * * * * * * * * * * * * * * *')
      res = check(val_dataloader, model, step, criterion, kwargs={'loss_weights':train_weights, 'mode': 'val'})
      print()
      if res > best_score:
        best_model = copy.deepcopy(model.state_dict())
        best_score = res
  
  # End of one epoch, update.
  optimizer.step()
  optimizer.zero_grad()

  model.step = step
  
  train_score = check(train4val_dataloader, model, step, criterion, kwargs={'loss_weights':train_weights, 'mode': 'train'})
  
  if e % save_every == 0:
    filename = '{}epochs.pkl'.format(total_e)
    savemodel(filename, model)
  
  return {
    'model': model,
    'best': {
      'score': best_score,
      'model': best_model
    },
    'train_score': train_score
  }

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

  # Weights: The number of samples in each class.
  # Train_weights: 1/Weights, with normalization.
  weights = dataloader['train'].dataset.weights
  train_weights = 1 / weights
  s = np.sum(train_weights)
  device = os.environ['device']
  if int(os.environ['gpus']) > 1:
    device=device[:device.find(',')]

  train_weights = torch.from_numpy(train_weights / s).to(device=device, dtype=torch.float32)

  info = {
      'since': since,
      'epochs': epochs,
      'train_weights': train_weights,
      'best':{
        'score': 0.0,
        'model': copy.deepcopy(model.state_dict())
      }
    }

  for e in range(epochs):
    model.fold = 0
    info['e'] = e
    res = train_one_epoch(model, dataloader, optimizer, criterion, info)
    model = res['model']
    best = res['best']
    info['best']['score'] = best['score']
    info['best']['model'] = copy.deepcopy(best['model'])

    # Save the best model after every epoch.
    torch.save({
        'state_dict': info['best']['model'],
        'step': str(model.step),
        'tb-logdir': os.environ['tb-logdir'],
        'epochs': str(model.epochs+e)
      },
      '{}fold{}/best.pkl'.format(os.environ['savepath'], model.fold)
    )

    writer.add_scalars('Aggragate/Score', {'Score': best['score']}, e+model.epochs)
    writer.add_scalars('Aggragate/Score', {'Train Score': res['train_score']}, e+model.epochs)

def train5folds(
  models,
  dataloaders,
  optimizers,
  criterion,
  epochs
):
  since = time.time()

  # Weights: The number of samples in each class.
  # Train_weights: 1/Weights, with normalization.
  weights = dataloaders[0]['train'].dataset.weights
  train_weights = 1 / weights
  s = np.sum(train_weights)
  device = os.environ['device']
  if int(os.environ['gpus']) > 1:
    device=device[:device.find(',')]

  train_weights = torch.from_numpy(train_weights / s).to(device=device, dtype=torch.float32)

  info = [None, None, None, None, None]
  for i in range(5):
    info[i] = {
      'since': since,
      'epochs': epochs,
      'train_weights': train_weights,
      'best':{
        'score': 0.0,
        'model': copy.deepcopy(models[i].state_dict())
      }
    }

  for e in range(epochs):
    mean_score = 0.0
    mean_train_score = 0.0
    for i in range(5):
      print ('Fold {}:'.format(i))
      models[i].fold = i
      info[i]['e'] = e

      models[i] = modelLoader.move(models[i])
      res = train_one_epoch(models[i], dataloaders[i], optimizers[i], criterion, info[i])
      models[i] = res['model']
      models[i] = modelLoader.moveback(models[i])
      release(res['model'])

      best = res['best']
      info[i]['best']['score'] = best['score']
      info[i]['best']['model'] = copy.deepcopy(best['model'])

      # Save a best model after every epoch.
      torch.save({
          'state_dict': info[i]['best']['model'],
          'step': str(models[i].step),
          'tb-logdir': os.environ['tb-logdir'],
          'epochs': str(models[i].epochs+e)
        },
        '{}fold{}/best.pkl'.format(os.environ['savepath'], i)
      )
      mean_score += best['score']/5
      mean_train_score += res['train_score']/5

    writer.add_scalars('CrossFolds/Score', {'Score': mean_score}, e+models[0].epochs)
    writer.add_scalars('CrossFolds/Score', {'Train Score': mean_train_score}, e+models[0].epochs)