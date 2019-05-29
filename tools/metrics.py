import sklearn.metrics as metrics
import numpy as np
import pandas as pd

def isic18(scores, y, labels):
  """
  Metrics of ISIC challenge 2018.

  Args:
    scores: Matrix of scores in shape (M, N), where M = #Test Samples, N = #Classes
    y: Vector of ground truth of test data.
    labels: List of label names.
  
  Return:
    scoring: The scoring matrix in the following order.
      
      AUC
      AUC, Sens > 80%
      Average Precision
      Accuracy
      Sensitivity
      Specificity
      Dice Coefficient
      PPV
      NPV
  """
  # Score -> prediction
  y_pred = np.argmax(scores, axis=1)
  print (scores.shape)
  print(y_pred.shape)
  # prediction -> prediction matrix, shape (M, N)
  # where M = #Test Samples, N = #Classes
  prediction_matrix = None
  for k in range(len(labels)):
    print(y_pred.shape)
    print(np.where(y_pred == k, 1, 0), np.where(y_pred == k, 1, 0).shape)
    p = np.array(np.where(y_pred == k, 1, 0))[:, None]
    if prediction_matrix is None:
      prediction_matrix = p
    else:
      prediction_matrix = np.hstack((prediction_matrix, p))

  # Compute AUC, AUC with sensitivity > 80% and Average Precision
  auc_vec = np.zeros(len(labels))
  auc_sens_vec = np.zeros(len(labels))
  avg_pre_vec = np.zeros(len(labels))
  for k in range(len(labels)):
    if np.sum(prediction_matrix[:, k]) == 0:
      auc_vec[k] = 0
      auc_sens_vec[k] = 0
      avg_pre_vec[k] = 0
      continue
    fpr, tpr, _ = metrics.roc_curve(prediction_matrix[:, k], scores[:, k])
    auc_vec[k] = metrics.auc(fpr, tpr)
    tpr_sens = tpr[np.where(tpr < 0.8)]
    fpr_sens = fpr[np.where(tpr < 0.8)]
    bias = 0
    for j in range(1, len(fpr_sens)):
      if fpr_sens[j] > fpr_sens[j-1]:
        bias += (fpr_sens[j] - fpr_sens[j-1]) * tpr_sens[j]
    auc_sens_vec[k] = auc_vec[k] - bias
    avg_pre_vec[k] = metrics.average_precision_score(prediction_matrix[:, k], scores[:, k])

  cmatrix = metrics.confusion_matrix(y, y_pred)
  cmatrix_ele = None
  # END OF INTEGRAL METRICS

  # Get element-wise confusion matrix in shape: (N, 2, 2)
  # where N = #labels.
  for k in range(len(labels)):
    mat = np.zeros((2,2))
    mat[0,0] = cmatrix[k,k]
    mat[0,1] = np.sum(cmatrix[k, k+1:]) + np.sum(cmatrix[k, 0:k])
    mat[1,0] = np.sum(cmatrix[k+1:, k]) + np.sum(cmatrix[0:k, k])
    mat[1,1] = np.sum(cmatrix[k+1:, k+1:]) + np.sum(cmatrix[0:k, 0:k])
    mat = mat[None, :, :]
    if cmatrix_ele is None:
      cmatrix_ele = mat
      print(cmatrix_ele.shape, mat.shape)
    else:
      print(cmatrix_ele.shape, mat.shape)
      cmatrix_ele = np.vstack((cmatrix_ele, mat))

  # Compute Accuracy, Sensitivity, Specificity, Dice Coefficient, PPV and NPV
  diag = cmatrix_ele.diagonal(axis1=2)
  pred = np.sum(cmatrix_ele, axis=2)
  cond = np.sum(cmatrix_ele, axis=1)
  ppv = (diag/pred)[:,0]
  npv = (diag/pred)[:,1]
  sens = (diag/cond)[:,0]
  spec = (diag/cond)[:,1]
  acc = np.sum(diag, axis=1)/np.sum(cmatrix_ele, axis=(1,2))
  dice = 2/(1/sens+1/ppv)
  # END OF THRESHOLD METRICS

  scoring = auc_vec[None, :]
  scoring = np.vstack((scoring, auc_sens_vec[None, :]))
  scoring = np.vstack((scoring, avg_pre_vec[None, :]))
  scoring = np.vstack((scoring, acc[None, :]))
  scoring = np.vstack((scoring, sens[None, :]))
  scoring = np.vstack((scoring, spec[None, :]))
  scoring = np.vstack((scoring, dice[None, :]))
  scoring = np.vstack((scoring, ppv[None, :]))
  scoring = np.vstack((scoring, npv[None, :]))

  scoring = np.hstack((np.mean(scoring, axis=1)[:, None], scoring))

  index = [
    'AUC',
    'AUC, Sens>80%',
    'Average Precision',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'Dice Coefficient',
    'PPV',
    'NPV'
  ]
  columns = np.hstack((['mean'], labels))
  scoring = pd.DataFrame(scoring, index=index, columns=columns)

  return scoring