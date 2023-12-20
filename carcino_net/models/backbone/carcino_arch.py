from torch import nn
import torch
from fastai.callback.hook import model_sizes, hook_outputs, dummy_eval, Hooks, Hook
import numpy as np
from fastai.torch_basics import ConvLayer, BatchNorm, SigmoidRange, ToTensorBase,\
    PixelShuffle_ICNR, apply_init, SequentialEx
from fastai.vision.models.unet import ResizeToOrig
from carcino_net.models.backbone.helper import SpatialPyramidPooling, SkipBlock, SKIP_TYPE
from typing import List, Tuple, Callable, Optional
from torchvision.models.resnet import resnet50
from fastai.layers import NormType


def _get_sz_change_idxs(sizes):
    """Get the indexes of the layers where the size of the activation changes."""

    feature_szs = [size[-1] for size in sizes]
    sz_chg_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    return sz_chg_idxs


class GenericUNet(SequentialEx):
    """Adapted from DynamicUnet.

    """

    @staticmethod
    def decoder_with_skips(dummy_tensor: torch.Tensor, *,
                           layer_hooks: Hooks | List[Hook],
                           size_change_indices_bot_to_top: List[int],
                           sizes: List,
                           blur: bool,
                           blur_final: bool,
                           norm_type: NormType,
                           skip_type: SKIP_TYPE,
                           skip_bottleneck: bool,
                           act_cls: Callable,
                           init: Callable,
                           out_layers: Optional[List],
                           **kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        """Helper function to create decoder with skip connection

        Args:
            dummy_tensor: dummy tensor output of previous layers
            layer_hooks:
            size_change_indices_bot_to_top:
            sizes:
            blur:
            blur_final:
            norm_type:
            skip_type:
            skip_bottleneck:
            act_cls:
            init:
            out_layers:
            **kwargs:

        Returns:

        """
        # if not given then create a new list.
        if out_layers is None:
            out_layers = []

        # start from the deepest level block
        for i, idx in enumerate(size_change_indices_bot_to_top):
            not_final = i != len(size_change_indices_bot_to_top) - 1
            # input of upsampling / encoding path target size
            up_in_c, x_in_c = int(dummy_tensor.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            # would be too memory intensive to add attention here
            # sa = self_attention and (i==len(size_change_indices)-3)

            skip_block = SkipBlock(up_in_c, x_in_c, layer_hooks[i], final_div=not_final, blur=do_blur,
                                   act_cls=act_cls, init=init, norm_type=norm_type, skip_type=skip_type,
                                   bottleneck=skip_bottleneck,
                                   **kwargs).eval()
            out_layers.append(skip_block)
            dummy_tensor = skip_block(dummy_tensor)

        return out_layers, dummy_tensor

    @staticmethod
    def get_mid_bridge(dummy_tensor: torch.Tensor, *,
                       num_input: int,
                       feature_map_size: int,
                       pool_sizes,
                       ppm_flatten: bool,
                       act_cls: Callable, norm_type: NormType,
                       init: Callable,
                       **kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        """add midconv and poolings

        Returns:

        """
        # ni = sizes[-1][1]
        # map_size = sizes[-1][2]
        # ref_size = x.shape[-2:]
        ppm = SpatialPyramidPooling(grid_size_list=pool_sizes, flatten=False, feature_map_size=feature_map_size,
                                    in_channels=num_input, out_channels=num_input)
        ppm_out_size = SpatialPyramidPooling.output_size(pool_sizes, num_input, ppm_flatten)
        reduction_size = num_input
        reduction = nn.Sequential(
            ConvLayer(ppm_out_size, reduction_size, act_cls=act_cls, norm_type=norm_type, **kwargs),
        )

        middle_conv = nn.Sequential(ConvLayer(num_input, num_input*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
                                    ConvLayer(num_input*2, num_input, act_cls=act_cls,
                                              norm_type=norm_type, **kwargs)).eval()
        apply_init(nn.Sequential(middle_conv), init)
        out_modules = [ppm, reduction, middle_conv]
        dummy_tensor = SequentialEx(*out_modules)(dummy_tensor) # middle_conv(dummy_tensor)
        # down path + middle
        # encoder, BatchNorm(ni), nn.ReLU()
        return out_modules, dummy_tensor

    @staticmethod
    def get_output_layer(dummy_tensor: torch.Tensor,
                         imsize: Tuple[int, ...],
                         ref_size: Tuple[int, ...],
                         num_input: int,
                         num_output: int,
                         act_cls: Callable,
                         norm_type: NormType,
                         y_range: Optional = None,
                         **convlayer_kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        # ni = x.shape[1]
        layers = []
        if imsize != ref_size:
            layers.append(PixelShuffle_ICNR(num_input, act_cls=act_cls, norm_type=norm_type))

        layers.append(ResizeToOrig())
        layers += [ConvLayer(num_input, num_output, ks=1, act_cls=None, norm_type=norm_type, **convlayer_kwargs)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        layers.append(ToTensorBase())
        dummy_tensor: torch.Tensor = SequentialEx(*layers)(dummy_tensor)
        return layers, dummy_tensor

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
                 **convlayer_kwargs):
        """For simplification, last_cross was removed for now as it is not in the scope of our study.

        Args:
            encoder:
            n_out:
            img_size:
            pool_sizes:
            ppm_flatten:
            blur:
            blur_final:
            y_range:
            act_cls:
            init: Initialization function. Note it assumes that the encoder is already initialized by itself. Therefore
                only: midconv and upsampling routes are initialized
            norm_type:
            skip_type:
            skip_bottleneck:
            **convlayer_kwargs:
        """
        self.skip_type = skip_type

        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        size_change_indices_bot_to_top = list(reversed(_get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in size_change_indices_bot_to_top], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        # channel size
        ni = sizes[-1][1]
        map_size = sizes[-1][2]

        bridge_layers, x = GenericUNet.get_mid_bridge(x, num_input=ni, feature_map_size=map_size,
                                                      pool_sizes=pool_sizes, ppm_flatten=ppm_flatten,
                                                      act_cls=act_cls, norm_type=norm_type,
                                                      init=init, **convlayer_kwargs)
        # ref_size = x.shape[-2:]

        # down path + middle
        layers = [encoder, BatchNorm(ni), nn.ReLU()] + bridge_layers

        # up path
        decoders, x = GenericUNet.decoder_with_skips(x,
                                                     layer_hooks=self.sfs,
                                                     size_change_indices_bot_to_top=size_change_indices_bot_to_top,
                                                     sizes=sizes, blur=blur, blur_final=blur_final, norm_type=norm_type,
                                                     skip_type=skip_type,
                                                     skip_bottleneck=skip_bottleneck, act_cls=act_cls,
                                                     init=init, out_layers=None)
        # add decoders to the list
        layers += decoders

        output, x = GenericUNet.get_output_layer(x,
                                                 imsize=imsize,
                                                 ref_size=sizes[0][-2:], num_input=x.shape[1],
                                                 num_output=n_out,
                                                 act_cls=act_cls, norm_type=norm_type, **convlayer_kwargs)

        # add output layers
        layers += output

        # moved to function creating blocks
        # apply_init(nn.Sequential(layers[3],), init)  # add layers[-2] -- for last_cross/residual learning

        super().__init__(*layers)

    @classmethod
    def build_carcino(cls,
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
                   ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final, norm_type=norm_type,
                   y_range=y_range, act_cls=act_cls, init=init, skip_type=skip_type, skip_bottleneck=skip_bottleneck,
                   **kwargs)
