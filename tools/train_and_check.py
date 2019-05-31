import torch
import os
import numpy as np
import pandas as pd
from config.globaltb import writer
import sklearn.metrics as metrics

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

    num_correct = (y_pred == y_true).sum()
    num_samples = y_pred.shape[0]
    acc = float(num_correct) / num_samples

    # The sklearn.metrics.confusion_matrix is transposed compared with
    #   the 'Confusion matrix' in our mind, which is, prediction in each
    #   row, and condition in each column. The sklearn.metrics.confusion_matrix
    #   is prediction in each column, and condition in each row.
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred).T
    TP = confusion_matrix.diagonal()
    Prediction = confusion_matrix.sum(axis=1) # row sum
    Condition = confusion_matrix.sum(axis=0) # column sum
    recall = TP / Condition
    precision = TP / Prediction
    df_class_precision_recall = pd.DataFrame(
      np.vstack((precision, recall)),
      index=np.array(['precision', 'recall']),
      columns=np.array(classes)
    )

    # Print result
    acc_prompt = 'Total Accuracy: \nGot %d / %d correct: %.2f%% \n' % (num_correct, num_samples, 100 * acc)
    print(acc_prompt)

    sample_weight = [1/loader.dataset.weights[i] for i in y_true]
    aggregate = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    balanced_acc_prompt = '\nBalanced Multiclass Accuracy: %.4f \n' % aggregate
    print (balanced_acc_prompt)

    conf_mat_prompt = '\nThe confusion matrix:\n'
    print (pd.DataFrame(confusion_matrix, index=classes, columns=classes))

    pre_recall_prompt = '\nPrecision and Recall of each calss:\n'
    print(pre_recall_prompt)
    print(df_class_precision_recall)

    writer.add_scalars('Train/Acc',{'Acc': acc}, step)
    for i in range(C):
      writer.add_scalars('Train/'+classes[i], {
        'Precision': precision[i],
        'Recall': recall[i]
      }, step)
    writer.add_scalars('Train/BalancedAcc', {'Acc': aggregate}, step)


def train(
  model,
  optimizer,
  train_dataloader,
  val_dataloader,
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

  for e in range(epochs):
    for t, (x, y) in enumerate(train_dataloader):
      model.train()
      x = x.to(device=device, dtype=torch.float32)
      y = y.to(device=device, dtype=torch.long)

      # Forward prop.
      scores = model(x)
      loss = torch.nn.functional.cross_entropy(scores, y, train_weights)

      writer.add_scalars('Train/Loss',{'loss': loss.item()}, step)
      step += 1

      # Back prop.
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (t+1) % print_every == 0:
        print ('* * * * * * * * * * * * * * * * * * * * * * * *')
        print('Epoch %d,Iteration %d (total epoch %d, iteration %d): loss = %.4f' % (e+1, t+1, e+1+pretrain_epochs, step, loss.item()))
        print ('* * * * * * * * * * * * * * * * * * * * * * * *')
        check(val_dataloader, model, step)
        print()

    if (e+1) % save_every == 0:
      savepath = os.environ['savepath']
      savefilepath = savepath + str(e+pretrain_epochs+1) + 'epochs.pkl'
      # Check if the savepath is valid.
      if not os.path.exists(savepath):
        os.mkdir(savepath)
        print ('Create dir', savepath)
      # Save the model.
      torch.save({
        'state_dict': model.state_dict(),
        'episodes': str(step),
        'tb-logdir': os.environ['tb-logdir']
        },
        savefilepath
      )
      print ('Model save as', savefilepath)