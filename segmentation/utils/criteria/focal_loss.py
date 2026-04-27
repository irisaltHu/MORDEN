import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, smoothing=1e-32, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        alpha = torch.tensor(self.alpha).to(inputs.device)

        pt = inputs

        loss = - alpha * targets * (1 - pt) ** self.gamma * torch.log(pt + self.smoothing) \
               - (1 - alpha) * (1 - targets) * pt ** self.gamma * torch.log(1 - pt + self.smoothing)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss
