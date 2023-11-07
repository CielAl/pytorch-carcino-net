from torch import nn
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        eps = torch.finfo(pred.dtype).eps
        intersection = torch.dot(pred.flatten(), target.flatten())
        union = torch.sum(pred) + torch.sum(target) + eps

        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice
