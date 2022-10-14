from turtle import forward
import numpy as np
import tqdm
import random

import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss
    
class PointWiseCEloss(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, pos_logits, neg_logits, mask=None):
        if mask is None:
            mask = 1
        else:
            assert mask.shape == pos_logits.shape
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
        ) / torch.sum(mask)
        return loss