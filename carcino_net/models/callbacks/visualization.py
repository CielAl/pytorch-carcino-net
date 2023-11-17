from typing import Optional, Dict, List, Tuple, Union, Any, Sequence, Callable
import os
import torch
import warnings
import pytorch_lightning as L
from pytorch_lightning.strategies import Strategy
from torch.distributed import group as dist_group
from lightning_fabric.utilities.apply_func import convert_to_tensors
from lightning_utilities.core.apply_func import apply_to_collection
import pickle
from carcino_net.dataset.dataclass import ModelOutput
from carcino_net.dataset.utils import file_part
from carcino_net.visualization import export_showcase, pred_to_label, to_instance_map
# import operator

DEFAULT_REDUCE_OP = list.__add__  # operator.add


class OutputWriter(L.callbacks.BasePredictionWriter):

    export_dir: Optional[str]
    target_idx: int

    def __init__(self, export_dir: Optional[str] = None, target_idx: int = -1):

        super().__init__(write_interval='batch')
        self.export_dir = export_dir
        self._init_export_dir()
        self.target_idx = target_idx

    @staticmethod
    def path_invalid(export_dir):
        """Check if export_dir is not a str. Only as simple sanitization.

        Args:
            export_dir:

        Returns:

        """
        return export_dir is None or not isinstance(export_dir, str)

    def _init_export_dir(self):
        """Create the export folder after validation. Raise a warning if not a valid str.

        Returns:

        """
        assert not OutputWriter.path_invalid(self.export_dir), f"export_dir is not set - ignore output"
        os.makedirs(self.export_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction: ModelOutput,
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Callbacks to override in BasePredictionWriter. Defines how to export batch-level output.

        Write the batch-level output of each device. Specify the dataloader_idx and batch_idx as well as the
        rank of device.

        Args:
            trainer:
            pl_module:
            prediction:
            batch_indices:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """
        # BCHW - [0., 1.]
        img = prediction['img']
        # B 1 H W [0., 1.]
        mask_gt = prediction['mask']
        # B num_class H W [0., 1.]
        scores = prediction['pred_prob']
        uris: List[str] = prediction['uri']
        # to B H W C
        img_np = img.detach().permute(0, 2, 3, 1).cpu().numpy()
        # from B H W to B H W C

        label_gt_np = mask_gt.detach().cpu().numpy()
        # todo probably use colormap + predicted labels for multiclass
        scores_np = scores.detach().cpu().permute(0, 2, 3, 1).numpy()  # [:, self.target_idx, :, :]

        pred_label_np = pred_to_label(scores_np, class_axis=-1)

        for i, m, s, fname in zip(img_np, label_gt_np, pred_label_np, uris):
            fpart = file_part(fname)
            dest = os.path.join(self.export_dir, f"{fpart}_mask.png")
            pred_inst = to_instance_map(s, cmap='tab20')
            gt_inst = to_instance_map(m, cmap='tab20')

            export_showcase(image=i, ground_truth_mask=gt_inst, pred_mask=pred_inst, dest_name=dest)



