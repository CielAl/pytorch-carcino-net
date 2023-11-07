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
from scipy import ndimage
from carcino_net.dataset.utils import file_part
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Mask Label Collation')
parser.add_argument('--mask_dir',
                    default='W:/SICAPv2_fix/masks',
                    help='path to masks')
parser.add_argument('--mask_ext',
                    default='.png',
                    help='mask ext')
parser.add_argument('--export_folder',
                    type=str,
                    default='W:/SICAPv2_fix/masks_binary',
                    help='where to export')
parser.add_argument('--thresh',
                    type=int,
                    default=4,
                    help='threshold of HG. pixel >= thresh --> 255')
opt, _ = parser.parse_known_args(argv)

if __name__ == '__main__':
    mask_list = glob.glob(os.path.join(opt.mask_dir, f'*{opt.mask_ext}'))

    os.makedirs(opt.export_folder, exist_ok=True)

    for mask_file in tqdm(mask_list):
        mask = cv2.imread(mask_file)
        mask = mask[:, :, 0]
        mask = mask >= opt.thresh
        mask = ndimage.binary_fill_holes(mask)
        mask = mask.astype(np.uint8)
        mask *= 255

        dest_name = os.path.join(opt.export_folder, f"{file_part(mask_file)}.png")
        cv2.imwrite(dest_name, mask)
