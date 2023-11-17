import pytorch_lightning as L
import pandas as pd
import os
from .utils import file_part
from .sicap import SicapData
from torch.utils.data import DataLoader
from carcino_net.dataset.base import TransformWrapper, AbstractDataset
from torchvision.transforms import ToTensor
from typing import Optional, Callable


class SicapDataModule(L.LightningDataModule):
    train_sheet: pd.DataFrame
    train_dataset: AbstractDataset

    val_sheet: pd.DataFrame
    val_dataset: AbstractDataset
    train_transforms: Optional[Callable]
    val_transforms: Optional[Callable]


    def __init__(self,
                 train_sheet_dir: str,
                 val_sheet_dir: str,
                 image_dir: str, mask_dir: str,
                 num_workers: int,
                 batch_size: int,
                 train_transforms: Optional[Callable] = None,
                 train_pair: Optional[bool] = False,
                 val_pair: Optional[bool] = False,
                 val_transforms: Optional[Callable] = None,
                 image_ext: str = '.jpg', mask_ext: str = '.png',
                 seed: int = 0):
        super().__init__()
        self.image_dir = image_dir
        self.image_ext = image_ext

        self.mask_dir = mask_dir
        self.mask_ext = mask_ext
        self.seed = seed

        self.train_sheet_dir = train_sheet_dir
        self.val_sheet_dir = val_sheet_dir

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_transforms = train_transforms if train_transforms is not None else ToTensor()
        self.train_pair = train_pair
        self.val_transforms = val_transforms if val_transforms is not None else ToTensor()
        self.val_pair = val_pair


    @staticmethod
    def _fnames_from_sheet(sheet: pd.DataFrame, folder: str, ext: str, col_name: str = 'image_name',
                           sort_flag: bool = False):
        fname_list = list(sheet[col_name])
        fpart_list = [file_part(x) for x in fname_list]

        fname_ext = [os.path.join(folder, f"{x}{ext}") for x in fpart_list]
        if sort_flag:
            fname_ext.sort()
        return fname_ext

    def setup(self, stage: str = '') -> None:
        self.train_sheet = pd.read_excel(self.train_sheet_dir)
        img_list_train = SicapDataModule._fnames_from_sheet(self.train_sheet, self.image_dir,
                                                            self.image_ext, sort_flag=False)
        mask_list_train = SicapDataModule._fnames_from_sheet(self.train_sheet, self.mask_dir,
                                                             self.mask_ext, sort_flag=False)
        train_dataset = SicapData(img_list_train, mask_list_train, self.mask_ext)
        self.train_dataset = TransformWrapper(train_dataset, transforms=self.train_transforms,
                                              paired=self.train_pair)

        self.val_sheet = pd.read_excel(self.val_sheet_dir)
        img_list_val = SicapDataModule._fnames_from_sheet(self.val_sheet, self.image_dir,
                                                          self.image_ext, sort_flag=False)
        mask_list_val = SicapDataModule._fnames_from_sheet(self.val_sheet, self.mask_dir,
                                                           self.mask_ext, sort_flag=False)
        val_dataset = SicapData(img_list_val, mask_list_val, self.mask_ext)
        self.val_dataset = TransformWrapper(val_dataset, self.val_transforms, paired=self.val_pair)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def predict_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False, pin_memory=True,
                          persistent_workers=self.num_workers > 0)
