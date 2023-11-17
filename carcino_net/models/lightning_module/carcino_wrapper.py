from .base import BaseLightningModule, PHASE_STR
import torchmetrics
from torch import nn
import numpy as np
from typing import Sequence
# from torchvision.ops import sigmoid_focal_loss
# from functools import partial
from carcino_net.models.loss import DiceLoss, focal_loss_factory
from carcino_net.dataset.dataclass import ModelInput, ModelOutput


softmax = nn.Softmax(dim=1)


class CarcinoLightning(BaseLightningModule):

    def __init__(self,
                 carcino_backbone: nn.Module,
                 focal_alpha: Sequence[float] | np.ndarray,
                 focal_gamma: float,
                 batch_size: int, lr: float,
                 num_classes: int,
                 weight_decay: float = 1e-4,
                 betas=(0.5, 0.99),
                 max_t: int = 90,
                 prog_bar: bool = True, next_line: bool = True):
        """

        Args:
            batch_size: batch_size
            lr: learning rate
            max_t: max_t for CosAnnealingScheduler to restart
            prog_bar: whether to log results in progress bars
            next_line: whether to print a new line after each validation epoch. This enables the default tqdm progress
                bar to retain the results of previous epochs in previous lines.
        """
        super().__init__(batch_size=batch_size, lr=lr, max_t=max_t, prog_bar=prog_bar, next_line=next_line,
                         weight_decay=weight_decay, betas=betas)
        self.model = carcino_backbone
        self.num_classes = num_classes

        self.loss_avg = torchmetrics.MeanMetric()
        self.dice_metric_meter = torchmetrics.classification.Dice(num_classes=self.num_classes)
        self.dice_coeff_dbg = torchmetrics.MeanMetric()
        self.f1_meter = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='none')

        self.criteria_dice = DiceLoss(num_classes=self.num_classes)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.criteria_focal = focal_loss_factory(alpha=self.focal_alpha, gamma=self.focal_gamma, )
        # self.criteria_focal = partial(sigmoid_focal_loss, reduction='mean',
        #                               alpha=self.focal_alpha, gamma=self.focal_gamma)

    def _reset_meters(self):
        self.loss_avg.reset()
        self.dice_metric_meter.reset()
        self.dice_coeff_dbg.reset()
        self.f1_meter.reset()

    def _log_on_final_batch_helper(self, phase_name: PHASE_STR):
        self.log_meter(f"{phase_name}_dice_score", self.dice_metric_meter, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_loss", self.loss_avg, logger=True, sync_dist=True)
        self.log_meter(f"{phase_name}_dbg_coeff", self.dice_coeff_dbg, logger=True, sync_dist=True)
        f1_all = self.f1_meter.compute()
        if phase_name != 'fit':
            for idx, class_result in enumerate(f1_all):
                self.log(f"{phase_name}_f1_class_{idx}", class_result, logger=True, sync_dist=True,
                         prog_bar=self.prog_bar, batch_size=self.batch_size, on_epoch=True, on_step=False)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch: ModelInput, phase_name: PHASE_STR):
        """Step function helper shared by training and validation steps which computes the logits and log the loss.

        Args:
            batch: batch data in format of NetInput
            phase_name: the name of current phase, i.e., train or validation, for purpose of loss logging.

        Returns:
            NetOutput containing loss, logits (final-layer output) and true labels.
        """
        # stacked view of original and augmented images
        images = batch['img']
        masks = batch['mask']  # .long()
        # obtain the projection
        # N x class x H x W
        logits = self(images)
        # use softmax score as input
        focal_loss = self.criteria_focal(logits, masks.squeeze().long())

        scores = softmax(logits)
        dice_loss = self.criteria_dice(logits, masks.long())
        loss_all = focal_loss + dice_loss

        self.dice_metric_meter.update(logits, masks.long())
        self.loss_avg.update(loss_all)
        self.dice_coeff_dbg.update(1 - dice_loss)
        self.f1_meter.update(logits, masks.long())
        uri = batch['uri']

        out = ModelOutput(loss=loss_all, logits=logits, mask=masks, uri=uri, pred_prob=scores, img=images)
        self.log_on_final_batch(phase_name)
        return out

    def training_step(self, batch: ModelInput, batch_idx):
        out = self._step(batch, 'fit')
        self.scheduler_step()
        return out

    def validation_step(self, batch: ModelInput, batch_idx):
        out = self._step(batch, 'validate')
        return out

    def on_train_epoch_end(self) -> None:
        self._reset_meters()

    def on_validation_epoch_end(self) -> None:
        self._reset_meters()
        self.print_newln()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, 'predict')
