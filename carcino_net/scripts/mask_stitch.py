"""Stitch back the output labeled mask.
"""
import argparse
import os
import sys
from carcino_net.visualization.stitch import Stitch, ImageData
from carcino_net.visualization.helper import group_by_slidename
from carcino_net.visualization.masks import to_instance_map
import glob
from tqdm import tqdm
import re
import imageio
import skimage
from matplotlib.pyplot import get_cmap

print(os.getcwd())
argv = sys.argv[1:]
fold=3
parser = argparse.ArgumentParser(description='Carcino Prediction')

# /tmp/ramdisk/

# 'Z:/UTAH/running_output/carcino_multi_showcase/visualization_4'
parser.add_argument('--tile_mask_dir', default='Z:/UTAH/running_output/'
                                               'carcino_multi_showcase/visualization_4',
                    help='Viz export location')


parser.add_argument('--export_folder', default='Z:/UTAH/running_output/stitch4',
                    help='Export dir')

parser.add_argument('--mask_ext', default='.tiff',
                    help='extension of mask files')
parser.add_argument('--patch_size', default=512,
                    help='size of tiles')
parser.add_argument('--downsample_factor', default=16,
                    help='downsample factor')
# ------------ dataloader
parser.add_argument('--num_workers', default=8, type=int,
                    help='number of cpus used for DataLoader')
parser.add_argument('--bg_color', default=[31, 119, 180], type=int,
                    nargs='+',
                    help='bg color')
parser.add_argument('--cmap', default='tab10', type=str,
                    help='color map applied')
opt, _ = parser.parse_known_args(argv)

pattern_column = r'(?<=xini_)\d+'
pattern_row = r'(?<=yini_)\d+'

if __name__ == '__main__':
    os.makedirs(opt.export_folder, exist_ok=True)
    mask_list_all = glob.glob(os.path.join(opt.tile_mask_dir, f'*{opt.mask_ext}'))
    slide_dict = group_by_slidename(mask_list_all)
    for slide, mask_list in tqdm(slide_dict.items()):
        all_col = [int(re.findall(pattern_column, x)[0]) for x in mask_list]
        all_row = [int(re.findall(pattern_row, x)[0]) for x in mask_list]

        max_right = (max(all_col) + opt.patch_size) // opt.downsample_factor
        max_bottom = (max(all_row) + opt.patch_size) // opt.downsample_factor
        stitch = Stitch.build((max_bottom, max_right), init_color=tuple(opt.bg_color))
        for idx, mask_name in enumerate(mask_list):
            col = int(re.findall(pattern_column, mask_name)[0]) // opt.downsample_factor
            row = int(re.findall(pattern_row, mask_name)[0]) // opt.downsample_factor
            mask_label = imageio.v2.imread(mask_name)
            mask_inst = to_instance_map(mask_label, opt.cmap)
            mask = skimage.util.img_as_ubyte(mask_inst).squeeze()
            mask = skimage.transform.rescale(mask, 1/opt.downsample_factor, order=0, channel_axis=-1)

            box = (col, row)
            stitch_size = (opt.patch_size // opt.downsample_factor, opt.patch_size // opt.downsample_factor)
            stitch.paste(mask, box, stitch_size)
        fp = os.path.join(opt.export_folder, f"{slide}.png")
        stitch.source.save(fp, format_name=None)