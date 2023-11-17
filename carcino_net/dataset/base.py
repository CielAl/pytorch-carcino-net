from abc import abstractmethod
from torch.utils.data import Dataset
from .dataclass import ModelInput
from typing import Callable

SET_MODEL_INPUT = set(ModelInput.__annotations__.keys())


class AbstractDataset(Dataset):

    @abstractmethod
    def fetch(self, idx) -> ModelInput:
        ...

    def __getitem__(self, idx) -> ModelInput:
        data = self.fetch(idx)
        assert isinstance(data, dict)
        assert set(data.keys()).issubset(SET_MODEL_INPUT)
        return data

    @abstractmethod
    def __len__(self):
        ...


class TransformWrapper(AbstractDataset):

    def __init__(self, dataset: AbstractDataset, transforms: Callable, paired: bool = False):
        self.dataset = dataset
        self.transforms = transforms
        self.paired = paired

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def transform_helper(transforms, paired, img, mask):
        if transforms is None:
            return img, mask
        if paired:
            out = transforms(image=img, mask=mask)
            img = out['image']
            mask = out['mask']
        else:
            img = transforms(img)
            mask = transforms(mask)
        return img, mask

    def fetch(self, idx):
        data = self.dataset.fetch(idx)
        if self.transforms is not None:
            img, mask = TransformWrapper.transform_helper(self.transforms, self.paired,
                                                          data['img'], data['mask'])
            data['img'] = img
            data['mask'] = mask

        return data
