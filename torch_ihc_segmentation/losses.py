import torch
import torch.nn as nn
from torchmetrics.collections import MetricCollection


class FocalLossLogits(nn.Module):
    """
    From https://github.com/CellProfiling/HPA-competition-solutions/blob/547d53aaca148fdb5f4585526ad7364dfa47967d/bestfitting/src/layers/loss.py#L8
    """

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.scaling_factor = 1
        self.scaling_factor = self.forward(torch.zeros(1), torch.ones(1))

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = (
            logit
            - logit * target
            + max_val
            + ((-max_val).exp() + (-logit - max_val).exp()).log()
        )

        invprobs = nn.functional.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean() / self.scaling_factor


class FBetaBCELoss(nn.Module):
    def __init__(self, beta=2, epsilon=1e-6):
        super().__init__()
        assert beta > 1

        self.beta = beta
        self.epsilon = epsilon
        self.scaling_factor = 1
        self.scaling_factor = self.forward(torch.ones(1) * 0.5, torch.ones(1))

    def forward(self, probs, target):
        beta_sq = self.beta ** 2
        bce = nn.functional.binary_cross_entropy(probs, target)

        tp_loss = (target * (1 - bce)).sum(axis=-1)
        fp_loss = ((1 - target) * bce).sum(axis=-1)

        enumerator = (1 + beta_sq) * tp_loss
        denominator = (beta_sq * target.sum(axis=-1)) + tp_loss + fp_loss

        enumerator += self.epsilon
        denominator += self.epsilon

        return (1 - (enumerator / denominator).mean()) / self.scaling_factor


class ScaledBCELoss(nn.BCELoss):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 1
        self.scaling_factor = self.forward(torch.ones(1) * 0.5, torch.ones(1))

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs) / self.scaling_factor
