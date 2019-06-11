import torch
import torch.nn as nn

class RecallWeightedCrossEntropy(nn.Module):
  def __init__(self, weights):
    super(RecallWeightedCrossEntropy, self).__init__()
    self.weights = weights

  def forward(self, scores, y):
    _, pred = scores.max(1)
    recall_weights = weights[pred]
    exp_scores = torch.exp(scores)
    softmax = exp_scores[:, pred].diagonal() / exp_scores.sum(1)
    cross_entropy = -torch.log(softmax)
    loss = torch.sum(recall_weights * cross_entropy)
    return loss

def recall_cross_entropy(scores, y, weight):
  Res = RecallWeightedCrossEntropy(weight)
  return Res(scores, y)