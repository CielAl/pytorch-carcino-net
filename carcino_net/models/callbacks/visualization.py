from typing import Optional, List, Callable
import os
import pytorch_lightning as L
from carcino_net.dataset.dataclass import ModelOutput
from carcino_net.dataset.utils import file_part
from carcino_net.visualization import export_showcase, pred_to_label, to_instance_map
import numpy as np
import imageio


class VisualizeWriter(L.callbacks.BasePredictionWriter):
    export_dir: Optional[str]
    write_showcase: bool
    write_label_mask: bool
    write_score_mask: bool

    NUM_FLAGS: int = 3

    def __init__(self, export_dir: Optional[str] = None,
                 write_showcase: bool = True,
                 write_label_mask: bool = False,
                 write_score_mask: bool = False,
                 cmap: str = 'tab20'):
        super().__init__(write_interval='batch')
        self.export_dir = export_dir
        self._init_export_dir()

        self.write_showcase = write_showcase
        self.write_label_mask = write_label_mask
        self.write_score_mask = write_score_mask
        self.cmap = cmap

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
        assert not VisualizeWriter.path_invalid(self.export_dir), f"export_dir is not set - ignore output"
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

        Note that if mask is
        not present, the dataset must create placeholder/pseodo values in the 'mask' field.

        Write the batch-level output of each device. Specify the dataloader_idx and batch_idx as well as the
        rank of device.

        Args:
            trainer: The trainer that invokes this callback. (handled by trainer)
            pl_module: The corresponding pytorch-lightning module. (handled by trainer)
            prediction: Prediction result. (handled by trainer)
            batch_indices: (handled by trainer)
            batch: batch data (handled by trainer)
            batch_idx: index of batch (handled by trainer)
            dataloader_idx: Index of dataloader, e.g., for DDP. (handled by trainer)

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

        for i, m, lb, s, fname in zip(img_np, label_gt_np, pred_label_np, scores_np, uris):
            fpart = file_part(fname)
            pred_inst = to_instance_map(lb, cmap=self.cmap)
            gt_inst = to_instance_map(m, cmap=self.cmap)

            dest_showcase = os.path.join(self.export_dir, f"{fpart}_showcase.png")
            VisualizeWriter.export_on_flag(self.write_showcase, export_showcase,
                                           image=i, ground_truth_mask=gt_inst, pred_mask=pred_inst,
                                           dest_name=dest_showcase)

            dest_label_mask = os.path.join(self.export_dir, f"{fpart}.tiff")
            VisualizeWriter.export_on_flag(self.write_label_mask, imageio.v3.imwrite,
                                           dest_label_mask, lb, dtype=np.int64)

            dest_score_mask = os.path.join(self.export_dir, f"{fpart}_score.npy")
            VisualizeWriter.export_on_flag(self.write_score_mask, np.save,
                                           dest_score_mask, s)

    @staticmethod
    def export_on_flag(flag: bool, func: Callable, *args, **kwargs):
        if not flag:
            return
        return func(*args, **kwargs)

    @staticmethod
    def to_bit_bools(n: int, num_bits_limit: int):
        """Convert int to a list of bool: from the most significant bit to the least significant bit

        The number of bits of n cannot exceed (larger than) num_bits_limit. If smaller, the list will be padded with
        False.
        """
        n = int(n)
        # assert isinstance(n, int)
        assert n.bit_length() <= num_bits_limit
        return [(n >> i) & 1 == 1 for i in range(num_bits_limit - 1, -1, -1)]

    @classmethod
    def build(cls,
              export_dir: Optional[str] = None,
              write_mode: int = 7,
              cmap: str = 'tab20'):
        assert write_mode.bit_length() <= VisualizeWriter.NUM_FLAGS
        write_flags = VisualizeWriter.to_bit_bools(write_mode, num_bits_limit=VisualizeWriter.NUM_FLAGS)
        write_flags = write_flags[-VisualizeWriter.NUM_FLAGS:]
        [write_showcase, write_label, write_score] = write_flags
        return cls(export_dir=export_dir, write_showcase=write_showcase, write_label_mask=write_label,
                   write_score_mask=write_score, cmap=cmap)
