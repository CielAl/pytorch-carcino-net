from torch import nn
from fastai.callback.hook import model_sizes, hook_outputs, dummy_eval
import numpy as np
from fastai.torch_basics import ConvLayer, BatchNorm, SigmoidRange, ToTensorBase,\
    PixelShuffle_ICNR, apply_init, SequentialEx
from fastai.vision.models.unet import ResizeToOrig
from carcino_net.models.backbone.helper import SpatialPyramidPooling, SkipBlock, SKIP_TYPE
from typing import List, Tuple
from torchvision.models.resnet import resnet50


def _get_sz_change_idxs(sizes):
    """Get the indexes of the layers where the size of the activation changes."""

    feature_szs = [size[-1] for size in sizes]
    sz_chg_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    return sz_chg_idxs


class CarcinoNet(SequentialEx):
    """Adapted from DynamicUnet.

    """
    def __init__(self,
                 encoder: nn.Sequential,
                 n_out: int,
                 img_size: Tuple[int, ...],
                 pool_sizes: List[int],
                 ppm_flatten: bool = False,
                 blur=False,
                 blur_final=True,
                 y_range=None,
                 act_cls=nn.ReLU,
                 init=nn.init.kaiming_normal_, norm_type=None,
                 skip_type: SKIP_TYPE = 'sum',
                 skip_bottleneck: bool = False,
                 **kwargs):
        self.skip_type = skip_type

        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        size_change_indices = list(reversed(_get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in size_change_indices], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        # channel size
        ni = sizes[-1][1]
        map_size = sizes[-1][2]
        # ref_size = x.shape[-2:]
        ppm = SpatialPyramidPooling(grid_size_list=pool_sizes, flatten=False, feature_map_size=map_size,
                                    in_channels=ni, out_channels=ni)
        ppm_out_size = SpatialPyramidPooling.output_size(pool_sizes, ni, ppm_flatten)
        reduction_size = ni
        reduction = nn.Sequential(
            ConvLayer(ppm_out_size, reduction_size, act_cls=act_cls, norm_type=norm_type, **kwargs),
        )

        middle_conv = nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
                                    ConvLayer(ni*2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, BatchNorm(ni), nn.ReLU(), ppm, reduction, middle_conv]

        # print(sizes)
        for i, idx in enumerate(size_change_indices):
            not_final = i != len(size_change_indices)-1
            # input of upsampling / encoding path target size
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            # would be too memory intensive to add attention here
            # sa = self_attention and (i==len(size_change_indices)-3)

            skip_block = SkipBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur,
                                   act_cls=act_cls, init=init, norm_type=norm_type, skip_type=skip_type,
                                   bottleneck=skip_bottleneck,
                                   **kwargs).eval()
            layers.append(skip_block)
            x = skip_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))

        layers.append(ResizeToOrig())

        layers += [ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)

        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        layers.append(ToTensorBase())
        super().__init__(*layers)

    @classmethod
    def build(cls,
              n_out: int,
              img_size: Tuple[int, ...],
              pool_sizes: List[int],
              ppm_flatten: bool = False,
              blur=False,
              blur_final=True,
              y_range=None,
              act_cls=nn.ReLU,
              init=nn.init.kaiming_normal_, norm_type=None,
              skip_type: SKIP_TYPE = 'sum',
              skip_bottleneck: bool = False,
              **kwargs):
        encoder = nn.Sequential(*list(resnet50().children())[:-2])
        return cls(encoder,
                   n_out=n_out, img_size=img_size, pool_sizes=pool_sizes,
                   ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final,
                   y_range=y_range, act_cls=act_cls, init=init, skip_type=skip_type, skip_bottleneck=skip_bottleneck,
                   **kwargs)
