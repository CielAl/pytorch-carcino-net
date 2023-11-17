from typing import Optional, Sequence
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    exclude_bg: bool
    apply_softmax: bool
    num_classes: int

    """
    Shape:
    - pred: BCHW
    - target: BHW - pixel labeled by the pixel value
    """

    def __init__(self, exclude_bg: bool = True, apply_softmax: bool = True, num_classes: int = -1):
        super().__init__()
        self.exclude_bg = exclude_bg
        self.apply_softmax = apply_softmax
        self.num_classes = num_classes

    @staticmethod
    def _channel_exclude_bg(target: torch.Tensor, exclude_bg: bool = True):
        """Assume BxCxd1x...xdk and BG is the first channel

        """
        if not exclude_bg:
            return target
        return target[:, 1:, ...]

    @staticmethod
    def _one_hot_encoding(target: torch.Tensor, num_classes: int, exclude_bg: bool = True):
        # shape dim x oh
        target = target.long()
        oh = F.one_hot(target, num_classes)
        oh = torch.einsum('bhwc -> bchw', oh)
        return DiceLoss._channel_exclude_bg(oh)

    @staticmethod
    def _score(logits: torch.Tensor, apply_softmax: bool = True, exclude_bg: bool = True):
        if not apply_softmax:
            return DiceLoss._channel_exclude_bg(logits, exclude_bg=exclude_bg)
        return DiceLoss._channel_exclude_bg(F.softmax(logits, dim=1), exclude_bg=exclude_bg)

    # noinspection PyMethodMayBeStatic
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        eps = torch.finfo(pred.dtype).eps
        # B H W (N_C - 1)
        target_oh = DiceLoss._one_hot_encoding(target, num_classes=self.num_classes, exclude_bg=self.exclude_bg).float()
        pred_score = DiceLoss._score(pred, apply_softmax=self.apply_softmax, exclude_bg=self.exclude_bg)

        intersection = torch.dot(pred_score.flatten(), target_oh.flatten())
        union = torch.sum(pred_score) + torch.sum(target_oh) + eps

        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice


class FocalLoss(nn.Module):
    """ Use Adeel Hassan's implementation: https://github.com/AdeelH/pytorch-multi-class-focal-loss

    Shape:
    - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
    - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss_factory(alpha: Optional[Sequence] = None,
                       gamma: float = 0.,
                       reduction: str = 'mean',
                       ignore_index: int = -100,
                       device='cpu',
                       dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl
