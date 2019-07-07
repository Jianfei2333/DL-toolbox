import torch
import torch.nn as nn

class DOCLoss(nn.Module):
  def __init__(self):
    super(DOCLoss, self).__init__()
    # self.weight = weight

  def forward(self, input, target, weight=None):
    sigmoid = 1 / (1 + torch.exp(-input))
    bias = torch.ones_like(input)
    bias[range(0, bias.shape[0]), target] = 0
    log_sigmoid = torch.log(bias - sigmoid)
    loss = -torch.sum(log_sigmoid)
    return loss.item()

  def prediction(input, t=0.5, weight=None):
    sigmoid = 1 / (1 + torch.exp(-input))
    values, indices = sigmoid.max(1)
    predict = torch.where(values > t, indices, torch.tensor([-1]))
    return predict