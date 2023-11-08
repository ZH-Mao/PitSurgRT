import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean', loss_weight = 1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        # inputs should be logits, i.e. raw output of the last layer of your model, not passed through any activation function like Sigmoid
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)*self.loss_weight
        elif self.reduce == 'sum':
            return torch.sum(F_loss)*self.loss_weight
        else: 
            return F_loss*self.loss_weight

if __name__ == "__main__":
    seed = 2
    torch.manual_seed(seed)
    x = torch.randn(2, 10, 2)
    x= torch.reshape(x, (2, -1))
    y = torch.randn(2, 10, 2)
    y = torch.reshape(y, (2, -1))
    loss = FocalLoss()
    print(loss(x, y))