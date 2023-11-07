from torchvision.transforms import ToTensor
from albumentations import BasicTransform


class PairedToTensor(BasicTransform):

    def __init__(self):
        super().__init__()
        self.transform = ToTensor()

    # noinspection PyMethodOverriding
    def __call__(self, *, image=None, mask=None):
        o_dict = dict()
        if image is not None:
            o_dict['image'] = self.transform(image)
        if mask is not None:
            o_dict['mask'] = self.transform(mask)
        return o_dict

