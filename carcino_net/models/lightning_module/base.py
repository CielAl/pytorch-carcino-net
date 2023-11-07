import pytorch_lightning as L
import torchmetrics
from typing import Literal, Tuple
from abc import abstractmethod
import torch


PHASE_TRAIN = Literal['fit']
PHASE_VAL = Literal['validate']
PHASE_TEST = Literal['test']
PHASE_PRED = Literal['predict']
PHASE_STR = Literal[PHASE_TRAIN, PHASE_VAL, PHASE_TEST, PHASE_PRED]


class BaseLightningModule(L.LightningModule):
    WARM_UP_EPOCH: int = 10

    lr: float
    batch_size: int
    prog_bar: bool
    # whether print new line after each epoch
    next_line: bool
    max_t: int

    def configure_optimizers(self):
        """Set up the optimizer and lr scheduler - adapted from https://theaisummer.com/simclr/


        Returns:
            See LightningModule for more detail. Usually it returns a single optimizer or
            Tuple[List[optimizer], List[scheduler]]/
        """
        def exclude_from_wd_and_adaptation(name):
            if 'bn' in name:
                return True

        param_groups = [
            {
                'params': [p for name, p in self.named_parameters() if not exclude_from_wd_and_adaptation(name)],
                'weight_decay': self.weight_decay,
                'layer_adaptation': True,
            },
            {
                'params': [p for name, p in self.named_parameters() if exclude_from_wd_and_adaptation(name)],
                'weight_decay': 0.,
                'layer_adaptation': False,
            },
        ]
        optimizer = torch.optim.Adam(param_groups, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_t, eta_min=0,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def log_meter(self, name: str, metric: torchmetrics.Metric, on_step: bool = False,
                  on_epoch: bool = True, sync_dist: bool = True,
                  logger: bool = True):
        value = metric.compute()
        self.log(name, value, on_step=on_step, on_epoch=on_epoch, prog_bar=self.prog_bar,
                 logger=logger, batch_size=self.batch_size, sync_dist=sync_dist)
        return value

    def scheduler_step(self) -> None:
        sch = self.lr_schedulers()
        # self.log("LR", sch.get_last_lr(), on_epoch=True, prog_bar=self.prog_bar,
        #          logger=True, batch_size=self.batch_size, sync_dist=True)
        if self.trainer.is_last_batch and self.trainer.current_epoch >= self.WARM_UP_EPOCH:
            sch.step()

    def __init__(self, batch_size: int, lr: float, max_t: int, prog_bar: bool, next_line: bool,
                 weight_decay: float, betas: Tuple[float, ...]):
        """

        Args:
            batch_size: batch_size
            lr: learning rate
            max_t: max_t for CosAnnealingScheduler to restart
            prog_bar: whether to log results in progress bars
            next_line: whether to print a new line after each validation epoch. This enables the default tqdm progress
                bar to retain the results of previous epochs in previous lines.
        """
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.prog_bar = prog_bar
        self.next_line = next_line
        self.max_t = max_t
        self.weight_decay = weight_decay
        self.betas = betas

    @abstractmethod
    def _reset_meters(self, *args, **kwargs):
        """reset all torchmetrics meters

        Args:
            *args:
            **kwargs:

        Returns:

        """
        ...

    def print_newln(self):
        if self.next_line:
            print("\n")

    def log_on_final_batch(self, phase_name: PHASE_STR):
        """Operations and measurements to log at the final batch, e.g., for epoch-level outputs.

        Override `_log_on_final_batch_helper` for detailed procedures.

        Args:
            phase_name:

        Returns:

        """
        if not self.trainer.is_last_batch:
            return
        self._log_on_final_batch_helper(phase_name)

    @abstractmethod
    def _log_on_final_batch_helper(self, phase_name: PHASE_STR):
        """Override to implement detailed procedures in log_on_final_batch.

        Args:
            phase_name:

        Returns:

        """
        ...
