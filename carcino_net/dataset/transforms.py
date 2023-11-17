from torchvision.transforms import ToTensor, PILToTensor
from albumentations import BasicTransform
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import ToFloat
from typing import Callable, Optional
import numpy as np

# apparently Albumentation's ToFloat is image-only-transformation for some reason.

# class PairedToFloat(Callable):
#
#     to_float_img: Callable
#     to_float_mask: Callable
#
#     @staticmethod
#     def _get_transform_out(tf: Callable, data: np.ndarray):
#         return tf(image=data)['image']
#
#     def __init__(self, max_image: Optional[float] = None, max_mask: Optional[float] = None):
#         super().__init__()
#         self.to_float_img = ToFloat(max_value=max_image)
#         self.to_float_mask = ToFloat(max_value=max_mask)
#
#     # noinspection PyMethodOverriding
#     def __call__(self, *, image=None, mask=None):
#         o_dict = dict()
#         if image is not None:
#             o_dict['image'] = PairedToFloat._get_transform_out(self.to_float_img, image)
#         if mask is not None:
#             o_dict['mask'] = PairedToFloat._get_transform_out(self.to_float_mask, mask)
#         return o_dict
#
