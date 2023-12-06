import numpy as np
from typing import Tuple, Optional, Any, Generic, TypeVar
import PIL
from PIL.Image import Image as PILImage
from abc import ABC, abstractmethod
import imageio

T = TypeVar('T')


class ImageWrapper(ABC, Generic[T]):

    im: T

    def __init__(self, im: T):
        self.im = im

    @property
    @abstractmethod
    def shape_hw(self):
        ...

    @abstractmethod
    def paste(self, box, stitch_size, tile):
        ...

    @abstractmethod
    def save(self, fp, format_name: str = None):
        ...


class ImageNumpy(ImageWrapper[np.ndarray]):

    @property
    def shape_hw(self):
        return self.im.shape

    def paste(self, box: Tuple[int, int], stitch_size: Tuple[int, int], tile: np.ndarray):
        left, top = box
        tile = np.array(tile, copy=False)
        stitch_height, stitch_width = stitch_size
        self.im[top: top + stitch_height, left: left + stitch_width] = tile
        return tile

    def __init__(self, im):
        im = np.array(im, copy=False)
        super().__init__(im)

    def save(self, fp: str, format_name: str = None):
        imageio.v2.imwrite(fp, self.im, format=format_name)


class ImagePIL(ImageWrapper[PILImage]):

    @property
    def shape_hw(self):
        return self.im.size[::-1]

    def paste(self, box: Tuple[int, int], stitch_size: Tuple[int, int], tile: np.ndarray):
        try:
            tile = PIL.Image.fromarray(tile)
            self.im.paste(tile, box)
            return tile
        except:
            breakpoint()

    def __init__(self, im):
        match im:
            case PILImage():
                ...
            case np.ndarray():
                im = PIL.Image.fromarray(im)
            case str():
                im = PIL.Image.open(im).convert("RGB")
            case _:
                raise TypeError(f"Unsupported im type: {type(im)}")
        super().__init__(im)

    def save(self, fp, format_name = None):
        self.im.save(fp, format=format_name)


class ImageData:

    def __new__(cls, im: Any):
        match im:
            case np.ndarray():
                return ImageNumpy(im)
            case PILImage():
                return ImagePIL(im)


class Stitch(ABC):
    source: ImageWrapper

    @staticmethod
    def validate_stitch_size(*, tile_shape_hw: Optional[Tuple[int, int]],
                             stitch_size_hw: Optional[Tuple[int, int]],
                             coord_range: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        """validate the stitch size.

        If stitch_size_hw is none then use the whole tile. If coord_range is given then the stitch size will be
        clipped if right/bottom of tile to stitch exceeds the boundary of source image.

        Args:
            tile_shape_hw: shape of tile to stitch in height/width
            stitch_size_hw:  size to stitch
            coord_range: left, top, right_most_open, bottom_most_open

        Returns:

        """
        if isinstance(stitch_size_hw, float):
            return int(stitch_size_hw), int(stitch_size_hw)
        if stitch_size_hw is None:
            assert tile_shape_hw is not None
            return tuple(tile_shape_hw)
        assert isinstance(stitch_size_hw, Tuple)
        if coord_range is not None:
            height, width = stitch_size_hw
            left, top, right_most_open, bottom_most_open = coord_range
            width = min(right_most_open - left, width)
            height = min(bottom_most_open - top, height)
            stitch_size_hw = (height, width)
        return stitch_size_hw

    @staticmethod
    def paste_from_tile(source: ImageWrapper,
                        tile: np.ndarray, box_left_top: Tuple[int, int], stitch_size: Optional[Tuple[int, int]]):
        """paste the tile to source.

        Args:
            source:
            tile:
            box_left_top: left, top
            stitch_size: stitch_height, stitch_width

        Returns:

        """

        bottom_most_open, right_most_open, *_ = source.shape_hw[:]
        coord_range = box_left_top + (right_most_open, bottom_most_open)
        stitch_size = Stitch.validate_stitch_size(tile_shape_hw=tile.shape,
                                                  stitch_size_hw=stitch_size, coord_range=coord_range)
        stitch_height, stitch_width = stitch_size
        tile = tile[:stitch_height, :stitch_width]
        source.paste(box_left_top, stitch_size, tile)

    def __init__(self, source: ImageWrapper, ):
        self.source = source

    def paste(self, tile: np.ndarray, box: Tuple[int, int], stitch_size: Optional[Tuple[int, int]] = None):
        Stitch.paste_from_tile(self.source, tile, box, stitch_size)

    @classmethod
    def build(cls, source_shape_hw: Tuple[int, int], init_color: float | Tuple[float, ...]):
        size_wh = source_shape_hw[::-1]
        source = PIL.Image.new(mode="RGB", size=size_wh, color=init_color)
        return cls(ImageData(source))

    @property
    def image(self):
        return self.source.im
