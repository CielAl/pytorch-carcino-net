from typing import List, Callable, Optional, Literal
import torch
from fastai.layers import PixelShuffle_ICNR, BatchNorm, ConvLayer, NormType
from fastai.torch_core import Module, apply_init
from torch import nn
from torch.nn import functional as F
from fastai.callback.hook import Hook, Hooks


SKIP_CAT = Literal['cat']
SKIP_SUM = Literal['sum']
SKIP_TYPE = Literal[SKIP_CAT, SKIP_SUM]


def valid_add(tensor1, tensor2):
    """Check if two  tensors can be added (e.g., same shape or broadcast)
    """
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # different dim and neither is one
    for dim1, dim2 in zip(shape1[::-1], shape2[::-1]):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    # different number of dimensions
    if len(shape1) != len(shape2):
        shorter, longer = sorted([shape1, shape2], key=len)
        if any(dim != 1 for dim in longer[:len(longer) - len(shorter)]):
            return False

    return True


class ResizeConv(nn.Module):
    def __init__(self,
                 new_size,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 activation: Callable = nn.ReLU):
        super().__init__()
        self.new_size = new_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        p = F.interpolate(input=x, size=self.new_size, mode='nearest')
        return self.conv(p)


class SpatialPyramidPooling(nn.Module):
    grid_size_list: List[int]
    flatten: bool
    upconv_dict: nn.ModuleDict
    _do_upsample: bool

    def __init__(self, grid_size_list: List[int], flatten: bool = True,
                 feature_map_size: Optional[int] = None,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None):
        super().__init__()
        assert len(grid_size_list) > 0
        self.grid_size_list = grid_size_list
        self.flatten = flatten
        self.upconv_dict = nn.ModuleDict()
        # placeholder
        self._identity = nn.Identity()
        self.create_upconv_blocks(self.grid_size_list, feature_map_size, in_channels, out_channels)

    @staticmethod
    def output_size(grid_size_list: List[int], num_channels: int, flatten: bool):
        if flatten:
            sum([x ** 2 for x in grid_size_list]) * num_channels
        return len(grid_size_list) * num_channels

    def get_upsample_block(self, grid_size) -> Optional[nn.Module]:
        key = str(grid_size)
        if key not in self.upconv_dict:
            return self._identity
        return self.upconv_dict[key]

    def create_upconv_blocks(self, grid_size_list, new_size, in_channels, out_channels):
        self._do_upsample = new_size is not None

        optional_arg_check = [new_size is None, in_channels is None, out_channels is None]
        assert all(optional_arg_check) or not any(optional_arg_check), \
            f"new_size, in_channels, out_channels must all be none or not-none"
        if not self._do_upsample:
            return
        for grid_size in grid_size_list:
            upconv = ResizeConv(new_size=new_size, kernel_size=1, in_channels=in_channels, out_channels=out_channels)
            self.upconv_dict[str(grid_size)] = upconv

    @staticmethod
    def _pool(x: torch.Tensor, grid_size: int, flatten: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        pooled_output = F.adaptive_avg_pool2d(x, output_size=grid_size)
        if flatten:
            pooled_output = pooled_output.view(batch_size, -1, 1, 1)
        return pooled_output

    @staticmethod
    def _upsample(x: torch.Tensor, block: Optional[nn.Module], do_upsample: bool):
        if do_upsample:
            assert block is not None
            return block(x)
        return x

    def forward(self, x) -> torch.Tensor | List[torch.Tensor]:
        pooled_outputs = []
        for grid_size in self.grid_size_list:
            # pool given the target grid size
            pooled_output = SpatialPyramidPooling._pool(x, grid_size, self.flatten)
            # upsample if needed
            upconv = self.get_upsample_block(grid_size)
            pooled_output = SpatialPyramidPooling._upsample(pooled_output, upconv, self._do_upsample)
            pooled_outputs.append(pooled_output)

        spp_output = torch.cat(pooled_outputs, dim=1)
        return spp_output


class SkipBlock(Module):

    @staticmethod
    def channel_size(up_in_c: int, x_in_c: int, final_div: bool, skip_type: SKIP_TYPE = 'sum'):
        match skip_type:
            case 'cat':
                ni = up_in_c // 2 + x_in_c
            case 'sum':
                ni = up_in_c // 2
            case _:
                raise ValueError(f"Invalid skip_type: {skip_type}")

        nf = ni if final_div else ni // 2
        return ni, nf

    def __init__(self,
                 up_in_c: int, x_in_c: int,
                 hook: Hook | Hooks | List[Hook],
                 final_div: bool = True, blur: bool = False,
                 act_cls: Callable = nn.ReLU,
                 init: Callable = nn.init.kaiming_normal_, norm_type: Optional[NormType] = None,
                 bottleneck: bool = False,
                 skip_type: SKIP_TYPE = 'sum',
                 **kwargs):
        super().__init__()
        self.skip_type = skip_type
        self.all_hook = hook
        self.pixel_shuffle = PixelShuffle_ICNR(up_in_c, up_in_c // 2 if final_div else up_in_c // 4,
                                               blur=blur, act_cls=act_cls, norm_type=norm_type)
        self.bn = BatchNorm(x_in_c)

        ni, nf = SkipBlock.channel_size(up_in_c, x_in_c, final_div, skip_type)
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=None, **kwargs)
        self.relu = act_cls()
        self.bottle_neck = bottleneck
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    @staticmethod
    def _skip_connect(tensor1: torch.Tensor, tensor2, skip_type: SKIP_TYPE = 'sum'):
        match skip_type:
            case 'sum':
                assert valid_add(tensor1, tensor2)
                return tensor1 + tensor2
            case 'cat':
                return torch.cat([tensor1, tensor2], dim=1)
            case _:
                raise ValueError(f"Invalid skip_type: {skip_type}")

    def align_hook_out(self):
        ...

    @staticmethod
    def hook_out_helper(hook: Hook):
        assert isinstance(hook, Hook)
        return hook.stored

    @staticmethod
    def hook_size_collation_inplace(tensor_list: List[torch.Tensor], flag: bool):
        """
        """
        if (not flag) or tensor_list is None or len(tensor_list) == 0:
            return tensor_list

        # HW
        ref_shape = tensor_list[0].shape[-2:]

        for idx, tensor in enumerate(tensor_list):
            if tensor.shape[-2:] != ref_shape:
                # Resize the tensor to match the reference shape
                tensor_list[idx] = F.interpolate(tensor, size=ref_shape, mode='bilinear', align_corners=False)

        return tensor_list

    @staticmethod
    def get_hook_out(hook: Hook | List[Hook] | Hooks, skip_type: Optional[SKIP_TYPE] = None) -> torch.Tensor:
        if isinstance(hook, Hook):
            return SkipBlock.hook_out_helper(hook)
        assert skip_type is not None
        # if a list is encountered, then reduce the outcome first
        out_list = [SkipBlock.hook_out_helper(x) for x in hook]
        # disabled for now - for debugging: check if the activation in hooks on the same level have the same HW
        out_list = SkipBlock.hook_size_collation_inplace(out_list, flag=False)
        match skip_type:
            case 'sum':
                return sum(out_list)
            case 'cat':
                return torch.cat(out_list, dim=1)
            case _:
                raise NotImplementedError(f"{skip_type}")

    def forward(self, up_in):
        hook_out = SkipBlock.get_hook_out(self.all_hook, self.skip_type)  # self.hook.stored
        up_out = self.pixel_shuffle(up_in)
        skip_shape = hook_out.shape[-2:]
        if skip_shape != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, hook_out.shape[-2:], mode='nearest')
        cat_x = SkipBlock._skip_connect(up_out, self.bn(hook_out), self.skip_type)
        cat_x = self.relu(cat_x)
        if self.bottle_neck:
            return self.conv2(self.conv1(cat_x))
        return cat_x
