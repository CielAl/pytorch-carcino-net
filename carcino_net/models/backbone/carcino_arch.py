from torch import nn
import torch
from fastai.callback.hook import model_sizes, hook_outputs, hook_output, dummy_eval, Hooks, Hook
import numpy as np
from fastai.torch_basics import ConvLayer, BatchNorm, SigmoidRange, ToTensorBase, \
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


class GenericUNet(nn.Module):
    """Adapted from DynamicUnet.

    """
    head_list: nn.ModuleList

    @staticmethod
    def decoder_with_skips(dummy_tensor: torch.Tensor, *,
                           layer_hooks: Hooks | List[Hook],
                           size_change_ind_rev: List[int],
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
            size_change_ind_rev:
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
        for i, idx in enumerate(size_change_ind_rev):
            not_final = i != len(size_change_ind_rev) - 1
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
        out_modules = [BatchNorm(num_input), nn.ReLU(), ppm, reduction, middle_conv]
        dummy_tensor = SequentialEx(*out_modules)(dummy_tensor)  # middle_conv(dummy_tensor)
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

    @staticmethod
    def parse_encoder(encoder: nn.Module, imsize: Tuple[int, ...])\
            -> Tuple[List[Tuple[int, ...]], List[int], Hooks | List[Hook], torch.Tensor]:
        sizes = model_sizes(encoder, size=imsize)
        size_change_ind_rev = list(reversed(_get_sz_change_idxs(sizes)))
        sfs = hook_outputs([encoder[i] for i in size_change_ind_rev], detach=False)
        x = dummy_eval(encoder, imsize).detach()
        return sizes, size_change_ind_rev, sfs, x

    @staticmethod
    def basic_unet(encoder: nn.Module,
                   *,
                   n_out: int,
                   img_size: Tuple[int, ...],
                   sizes: List[Tuple[int, ...]],
                   size_change_ind_rev: List[int],
                   hooks: Hooks | List[Hooks] | List[List[Hook]],
                   dummy_tensor: torch.Tensor,
                   pool_sizes: List[int],
                   ppm_flatten: bool,
                   blur: bool = False,
                   blur_final: bool = True,
                   y_range=None,
                   act_cls: Callable,
                   norm_type: NormType,
                   init: Callable,
                   skip_type: SKIP_TYPE = 'sum',
                   skip_bottleneck: bool = False,
                   **convlayer_kwargs) -> Tuple[Tuple[List[nn.Module], List[nn.Module],
                                                List[nn.Module], List[nn.Module]], torch.Tensor]:
        # channel size
        ni = sizes[-1][1]
        map_size = sizes[-1][2]

        down_path = [encoder]   # BatchNorm(ni), nn.ReLU() moved to middle layers.

        bridge_layers, dummy_tensor = GenericUNet.get_mid_bridge(dummy_tensor, num_input=ni, feature_map_size=map_size,
                                                                 pool_sizes=pool_sizes, ppm_flatten=ppm_flatten,
                                                                 act_cls=act_cls, norm_type=norm_type,
                                                                 init=init, **convlayer_kwargs)
        # ref_size = x.shape[-2:]

        # down path + middle

        # up path
        decoders, dummy_tensor = GenericUNet.decoder_with_skips(dummy_tensor,
                                                                layer_hooks=hooks,
                                                                size_change_ind_rev=size_change_ind_rev,
                                                                sizes=sizes, blur=blur, blur_final=blur_final,
                                                                norm_type=norm_type,
                                                                skip_type=skip_type,
                                                                skip_bottleneck=skip_bottleneck, act_cls=act_cls,
                                                                init=init, out_layers=None)
        # add decoders to the list

        output, dummy_tensor = GenericUNet.get_output_layer(dummy_tensor,
                                                            imsize=img_size,
                                                            ref_size=sizes[0][-2:], num_input=dummy_tensor.shape[1],
                                                            num_output=n_out,
                                                            act_cls=act_cls, norm_type=norm_type,
                                                            y_range=y_range, **convlayer_kwargs)

        # add output layers

        return (down_path, bridge_layers, decoders, output), dummy_tensor

    @staticmethod
    def indice_to_slices(*, size_change_ind_forward: List[int]) -> List[Tuple[int, int]]:
        size_change_ind_forward = np.asarray(size_change_ind_forward, dtype=int)
        size_change_ind_forward += 1
        size_change_ind_padded = np.pad(size_change_ind_forward, pad_width=(1, 0))
        return [(int(size_change_ind_padded[idx - 1]), int(size_change_ind_padded[idx]))
                for idx in range(1, len(size_change_ind_padded))]

    @staticmethod
    def slice_backbone_helper(backbone: nn.Sequential, slice_idx: List[Tuple[int, int]])\
            -> List[nn.Module]:
        out_modules = []
        # out_heads = nn.ModuleList()
        for slice_pair in slice_idx:
            start, end = slice_pair
            out_modules.append(backbone[start: end])
        return out_modules

    @staticmethod
    def slice_backbone(backbone: nn.Sequential, size_change_ind_forward: List[int]) -> List[nn.Module]:
        slice_idx = GenericUNet.indice_to_slices(size_change_ind_forward=size_change_ind_forward)
        return GenericUNet.slice_backbone_helper(backbone, slice_idx)

    @staticmethod
    def expand_hooks_inplace(decoders: nn.Sequential | List[nn.Module], hooks: List[List[Hook]]):
        assert len(decoders) <= len(hooks)
        for d, hk in zip(decoders, hooks):
            hk: List[Hook]
            new_hook = hook_output(d, detach=False)
            hk.append(new_hook)
        return hooks

    @staticmethod
    def get_decoder_head(curr_change_ind_rev: List[int], *, curr_sizes: List[Tuple[int, ...]],
                         curr_hooks: List[List[Hook]],
                         curr_encoder: nn.Sequential,
                         n_out: int,
                         img_size: Tuple[int, int],
                         dummy_tensor_in: torch.Tensor,
                         pool_sizes: List[int],
                         ppm_flatten: bool,
                         blur: bool,
                         blur_final: bool,
                         y_range,
                         act_cls: Callable,
                         norm_type: NormType,
                         skip_type: SKIP_TYPE,
                         init: Callable,
                         skip_bottleneck: bool,
                         **convlayer_kwargs) -> nn.Sequential:

        # get the unet layers
        layer_tuple, d_out = GenericUNet.basic_unet(curr_encoder, n_out=n_out, img_size=img_size,
                                                    sizes=curr_sizes, size_change_ind_rev=curr_change_ind_rev,
                                                    hooks=curr_hooks, dummy_tensor=dummy_tensor_in,
                                                    pool_sizes=pool_sizes, ppm_flatten=ppm_flatten, blur=blur,
                                                    blur_final=blur_final, y_range=y_range, act_cls=act_cls,
                                                    norm_type=norm_type, skip_type=skip_type, init=init,
                                                    skip_bottleneck=skip_bottleneck, **convlayer_kwargs)

        # down is essentially the current piece of encoder
        down, bridge, decoder, out_layer = layer_tuple
        # expand the hook -- add decoder to the current hooks
        GenericUNet.expand_hooks_inplace(decoder, curr_hooks)
        concatenated_decoders = bridge + decoder + out_layer
        return nn.Sequential(*concatenated_decoders)

    @staticmethod
    def nested_unet(encoder: nn.Sequential,
                    num_levels: int = 0,
                    *,
                    n_out: int,
                    img_size: Tuple[int, int],
                    sizes: List[Tuple[int, ...]],
                    size_change_ind_rev: List[int],
                    hooks: Hooks | List[Hook],
                    dummy_tensor_in: torch.Tensor,
                    pool_sizes: List[int],
                    ppm_flatten: bool,
                    blur: bool = False,
                    blur_final: bool = True,
                    y_range=None,
                    act_cls: Callable,
                    norm_type: NormType,
                    init: Callable,
                    skip_type: SKIP_TYPE = 'sum',
                    skip_bottleneck: bool = False,
                    **convlayer_kwargs):
        # if nest_level = 0 ---> only create the out-most Unet
        # after appending each level of inner unet, append the output/decoder nodes to inner
        # hook list for the next level
        hook_stack: List[List[Hook]] = [[hook] for hook in hooks]
        backbone_slices: List = GenericUNet.slice_backbone(encoder, size_change_ind_forward=size_change_ind_rev[::-1])

        # regardless if it is nested, the encode_heads contains all slice of the backbones
        # encode_heads = nn.ModuleList([bs for bs in backbone_slices])

        num_levels = min(len(size_change_ind_rev) - 1, num_levels)
        #
        head_list = nn.ModuleList()
        for idx, level in enumerate(range(1, num_levels + 1)):
            # start indices of the size_change_ind_rev for the current inner unet
            start_ind = len(size_change_ind_rev) - level
            curr_change_ind_rev = size_change_ind_rev[start_ind::]

            # not reverse
            curr_sizes = sizes[:level]
            curr_hooks = hook_stack[:level]

            curr_end_ind = size_change_ind_rev[start_ind] + 1
            curr_encoder = encoder[:curr_end_ind]
            # get the unet layers
            encoder_head = backbone_slices.pop(0)
            decoder_head = GenericUNet.get_decoder_head(curr_change_ind_rev, curr_sizes=curr_sizes,
                                                        curr_hooks=curr_hooks,
                                                        curr_encoder=curr_encoder,
                                                        n_out=n_out,
                                                        img_size=img_size, dummy_tensor_in=dummy_tensor_in,
                                                        pool_sizes=pool_sizes,
                                                        ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final,
                                                        y_range=y_range, act_cls=act_cls, norm_type=norm_type,
                                                        skip_type=skip_type, init=init,
                                                        skip_bottleneck=skip_bottleneck, **convlayer_kwargs)
            head_list.append(nn.Sequential(encoder_head, decoder_head))
        # add the outer
        final_head = GenericUNet.get_decoder_head(size_change_ind_rev, curr_sizes=sizes, curr_hooks=hook_stack,
                                                  curr_encoder=encoder, n_out=n_out, img_size=img_size,
                                                  dummy_tensor_in=dummy_tensor_in, pool_sizes=pool_sizes,
                                                  ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final,
                                                  y_range=y_range,
                                                  act_cls=act_cls, norm_type=norm_type, skip_type=skip_type,
                                                  skip_bottleneck=skip_bottleneck, **convlayer_kwargs)
        # rest of the path as encoder
        head_list.append(nn.Sequential(*backbone_slices, final_head))
        return head_list

    def __init__(self,
                 encoder: nn.Sequential,
                 *,
                 n_out: int,
                 num_levels: int = 0,
                 img_size: Tuple[int, int],
                 pool_sizes: List[int],
                 ppm_flatten: bool = False,
                 blur: bool = False,
                 blur_final: bool = True,
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
        super().__init__()
        self.skip_type = skip_type

        sizes, size_change_ind_rev, self.sfs, x = GenericUNet.parse_encoder(encoder, img_size)

        head_list = GenericUNet.nested_unet(encoder, num_levels=num_levels,
                                            n_out=n_out, img_size=img_size,
                                            sizes=sizes,
                                            size_change_ind_rev=size_change_ind_rev, hooks=self.sfs,
                                            dummy_tensor_in=x, pool_sizes=pool_sizes,
                                            ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final,
                                            y_range=y_range, act_cls=act_cls, norm_type=norm_type,
                                            skip_type=skip_type, init=init,
                                            skip_bottleneck=skip_bottleneck, **convlayer_kwargs)
        self.head_list = head_list
        self.num_levels = num_levels

    @classmethod
    def build_carcino(cls,
                      n_out: int,
                      img_size: Tuple[int, int],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        out = []
        for head in self.head_list:
            out.append(head(x))
        if len(out) == 1:
            return out[0]
        return out
