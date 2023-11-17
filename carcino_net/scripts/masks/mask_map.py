"""
To verify the mapping of class label to pixel values.
"""
from typing import Tuple, List
import sys
import argparse
import cv2
import os
import glob
import json
import numpy as np
from tqdm import tqdm

from carcino_net.dataset.utils import file_part, mask_pixel_map

argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Mask Label Collation')
parser.add_argument('--mask_dir',
                    default='/tmp/ramdisk/SICAPV2Fixed/masks',
                    help='path to masks')
parser.add_argument('--mask_ext',
                    default='.png',
                    help='mask ext')
parser.add_argument('--export_folder',
                    type=str,
                    default='/tmp/ramdisk/SICAPV2Fixed/masks_mul',
                    help='where to export')
parser.add_argument('--map',
                    type=str,
                    default='./data/label_map.json',
                    help='location of pixel label maps')
opt, _ = parser.parse_known_args(argv)

if __name__ == '__main__':
    with open(opt.map, 'r') as root:
        label_dict = json.load(root)
        label_dict = {int(k): int(v) for k, v in label_dict.items()}
    mask_list = glob.glob(os.path.join(opt.mask_dir, f'*{opt.mask_ext}'))

    os.makedirs(opt.export_folder, exist_ok=True)

    for idx, mask_file in enumerate(tqdm(mask_list)):
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        mask_mapped = mask_pixel_map(mask, label_dict, label_dict)
        pix, count = np.unique(mask_mapped, return_counts=True)
        if not set(pix).issubset({0, 1, 2}):
            print(idx, mask_file)
            break
        dest_name = os.path.join(opt.export_folder, f"{file_part(mask_file)}.png")
        cv2.imwrite(dest_name, mask)
