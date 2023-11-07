from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
from .dataclass import ModelInput
from .utils import file_part
from .base import AbstractDataset


class SicapData(AbstractDataset):

    def __init__(self, img_list, mask_list, mask_ext):
        self.img_list = np.asarray(img_list)
        self.img_list.sort()
        self.mask_list = np.asarray(mask_list)
        self.mask_list.sort()
        self.mask_ext = mask_ext

    def __len__(self):
        return len(self.img_list)

    def fetch(self, idx) -> ModelInput:
        img_file = self.img_list[idx]
        mask_file = self.mask_list[idx]
        assert file_part(img_file) == file_part(mask_file)
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # todo add label if needed
        return ModelInput(img=img, uri=img_file, mask=mask, label=-1)

    def __getitem__(self, idx) -> ModelInput:
        return self.fetch(idx)
